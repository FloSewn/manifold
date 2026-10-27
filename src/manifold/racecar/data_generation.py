import os
from typing import Any

import cv2
import gymnasium as gym
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from gymnasium import spaces
from IPython.display import HTML
from matplotlib.figure import Figure


def update_physics(
    x: float, y: float, theta: float, velocity: float, action_steer: float, action_accel: float
) -> tuple[float, float, float, float]:
    friction: float = 0.95
    max_speed: float = 10.0

    # 1. Geschwindigkeit updaten (Reibung + Gas/Bremse)
    velocity = velocity * friction + action_accel
    velocity = max(min(velocity, max_speed), -max_speed)

    # 2. Lenkung anwenden (proportional zur Geschwindigkeit)
    theta += action_steer * (velocity * 0.1)

    # 3. Neue Position berechnen (Trigonometrie)
    x += velocity * float(np.cos(theta))
    y += velocity * float(np.sin(theta))

    return x, y, theta, velocity


class RaceEnv(gym.Env):
    """Custom Environment following the gym interface with strict typing"""

    metadata: dict[str, Any] = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(self) -> None:
        super(RaceEnv, self).__init__()

        # Aktionen: [Lenkung (-1 bis 1), Beschleunigung (-1 bis 1)]
        self.action_space: spaces.Box = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # Beobachtung: 64x64 RGB Bild (für das Vision Model / VAE)
        self.observation_space: spaces.Box = spaces.Box(
            low=0, high=255, shape=(64, 64, 3), dtype=np.uint8
        )

        # Track Layout
        self.track_radius: float = 20.0
        self.track_center: tuple[float, float] = (32.0, 32.0)

        # Initialisierung der State-Variablen für den Type Checker
        self.car_x: float = 0.0
        self.car_y: float = 0.0
        self.car_theta: float = 0.0
        self.car_v: float = 0.0
        self.steps: int = 0

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)

        alpha: float = float(np.random.uniform(0, 2 * np.pi))

        # --- NEU: Spawn-Radius stark variieren ---
        # Die Strecke hat Radius 20. Wir spawnen das Auto zufällig irgendwo
        # zwischen Radius 10 (weit innen) und 30 (weit außen).
        spawn_radius = self.track_radius + np.random.uniform(-10.0, 10.0)

        self.car_x = float(self.track_center[0] + spawn_radius * np.cos(alpha))
        self.car_y = float(self.track_center[1] + spawn_radius * np.sin(alpha))

        tangent_angle: float = alpha + np.pi / 2.0

        # --- NEU: Extreme Winkel zulassen ---
        # Manchmal steht das Auto beim Start im 90-Grad Winkel zur Strecke!
        self.car_theta = float(tangent_angle + np.random.uniform(-1.5, 1.5))
        self.car_v = float(np.random.uniform(0.0, 5.0))
        self.steps = 0

        obs: np.ndarray = self._get_obs()
        info: dict[str, Any] = {}

        return obs, info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self.steps += 1

        # Explizite Casts für sauberes Type Hinting und numpy-Extraktion
        steer: float = float(action[0])
        accel: float = float(action[1])

        # 1. Physik updaten
        self.car_x, self.car_y, self.car_theta, self.car_v = update_physics(
            self.car_x, self.car_y, self.car_theta, self.car_v, steer, accel
        )

        # 2. Reward berechnen
        dist_to_center: float = float(
            np.sqrt(
                (self.car_x - self.track_center[0]) ** 2 + (self.car_y - self.track_center[1]) ** 2
            )
        )
        error: float = abs(dist_to_center - self.track_radius)

        # Stärkere Belohnung für Geschwindigkeit, Strafe für Stillstand
        reward: float = self.car_v * 0.5 - error * 0.2 - 0.1

        # 3. Abbruchbedingung
        terminated: bool = bool(error > 15.0)
        truncated: bool = bool(self.steps > 1000)

        # Massive Strafe für das Verlassen der Strecke!
        if terminated:
            reward -= 100.0

        obs: np.ndarray = self._get_obs()
        info: dict[str, Any] = {}

        return obs, reward, terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        # Erzeuge ein leeres 64x64 schwarzes Bild
        img: np.ndarray = np.full((64, 64, 3), 255, dtype=np.uint8)

        # Strecke zeichnen (grauer Kreis)
        cv2.circle(
            img,
            (int(self.track_center[0]), int(self.track_center[1])),
            int(self.track_radius),
            (80, 80, 80),
            5,
        )

        # Auto zeichnen (rotes Rechteck, rotiert)
        rect: tuple[tuple[float, float], tuple[float, float], float] = (
            (self.car_x, self.car_y),
            (4.0, 2.0),
            float(np.degrees(self.car_theta)),
        )

        box: np.ndarray = cv2.boxPoints(rect)
        box_int: np.ndarray = np.int32(box)  # np.int32 ersetzt das veraltete np.int0

        cv2.drawContours(img, [box_int], 0, (255, 0, 0), -1)

        return img


def get_teacher_action(
    car_x: float,
    car_y: float,
    car_theta: float,
    track_center: tuple[float, float],
    track_radius: float = 20.0,
    epsilon: float = 0.0,
) -> np.ndarray:
    # 1. Wo ist das Auto auf dem Kreis?
    angle_from_center = float(np.arctan2(car_y - track_center[1], car_x - track_center[0]))

    # 2. Den Zielpunkt (die "Karotte") berechnen
    lookahead_angle = angle_from_center + 0.5
    target_x = track_center[0] + track_radius * np.cos(lookahead_angle)
    target_y = track_center[1] + track_radius * np.sin(lookahead_angle)

    # 3. ideal_heading und Fehler berechnen
    ideal_heading = float(np.arctan2(target_y - car_y, target_x - car_x))
    steer_error = ideal_heading - car_theta
    steer_error = (steer_error + np.pi) % (2 * np.pi) - np.pi

    # --- DER DRUNKEN TEACHER FIX ---
    # Wenn wir "glitchen" (Exploration), lenken wir absichtlich komplett in die falsche Richtung,
    # aber nur für einen kurzen Moment. Danach greift wieder der P-Regler, um das Auto zu retten.
    if np.random.rand() < epsilon:
        # Extreme, aber fließende Störung (kein reines Rauschen, sondern "falsch abbiegen")
        steer_action = np.random.choice([-1.0, 1.0])
    else:
        # P-Regler mit erhöhtem Rauschen, damit das Fahren organischer wirkt
        steer_action = steer_error * 2.5 + np.random.normal(0, 0.2)

    steer_action = np.clip(steer_action, -1.0, 1.0)
    accel_action = np.clip(np.random.normal(0.6, 0.2), -1.0, 1.0)

    return np.array([steer_action, accel_action], dtype=np.float32)


def generate_random_rollouts(
    env: gym.Env,
    total_episodes: int = 10000,
    max_steps_per_episode: int = 300,
    batch_size: int = 100,
    save_dir: str = "/content/drive/WorldModels/RaceCar/data",  # Typischer Colab-Pfad
) -> None:
    os.makedirs(save_dir, exist_ok=True)

    # Temporäre Listen für den aktuellen Batch (um RAM zu sparen und I/O zu optimieren)
    batch_obs: list[np.ndarray] = []
    batch_actions: list[np.ndarray] = []

    print(f"Starte Datengenerierung für {total_episodes} Episoden...")

    for ep in range(total_episodes):
        obs, info = env.reset()

        for step in range(max_steps_per_episode):
            # 1. Zufällige Aktion aus dem action_space ziehen (z.B. [0.4, -0.9])
            # action: np.ndarray = env.action_space.sample()
            action = get_teacher_action(
                env.car_x, env.car_y, env.car_theta, env.track_center, env.track_radius, epsilon=0.2
            )

            # 2. Spiel einen Schritt weiterlaufen lassen
            next_obs, reward, terminated, truncated, info = env.step(action)

            # 3. Daten speichern (Wichtig: Wir speichern das Bild VOR der Aktion)
            batch_obs.append(obs)
            batch_actions.append(action)

            obs = next_obs

            # Wenn das Auto von der Strecke fliegt, beenden wir die Episode frühzeitig
            if terminated or truncated:
                break

        # Batch auf der Festplatte speichern, wenn er voll ist
        if (ep + 1) % batch_size == 0:
            batch_index: int = (ep + 1) // batch_size

            # Listen in effiziente Numpy-Arrays umwandeln
            obs_array: np.ndarray = np.array(batch_obs, dtype=np.uint8)
            action_array: np.ndarray = np.array(batch_actions, dtype=np.float32)

            file_path: str = os.path.join(save_dir, f"rollouts_batch_{batch_index:03d}.npz")

            # Komprimiert speichern (.npz)
            np.savez_compressed(file_path, obs=obs_array, actions=action_array)
            print(f"Batch {batch_index} gespeichert: {len(obs_array)} Frames in {file_path}")

            # RAM wieder freimachen für den nächsten Batch
            batch_obs.clear()
            batch_actions.clear()


def load_and_animate_rollouts(file_path: str, fps: int = 30) -> HTML:
    # 1. Daten aus dem .npz Archiv laden
    data: np.lib.npyio.NpzFile = np.load(file_path)
    observations: np.ndarray = data["obs"]  # Shape: (N, 64, 64, 3)
    actions: np.ndarray = data["actions"]  # Shape: (N, 2)

    print(f"Lese Datei: {file_path}")
    print(f"Anzahl Frames: {observations.shape[0]}")

    # 2. Matplotlib Figure initialisieren
    fig: plt.Figure
    ax: plt.Axes
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.axis("off")  # Keine Achsen anzeigen, sieht aus wie ein echtes Game-Fenster

    # Erstes Bild in den Plot laden
    im = ax.imshow(observations[0])

    # Titel für die Anzeige der aktuellen Aktion
    title_text = ax.set_title("Frame 0 | Action: Steer=0.00, Accel=0.00", fontsize=10)

    # 3. Update-Funktion für die Animation
    def update(frame_idx: int) -> tuple[Any, Any]:
        im.set_array(observations[frame_idx])
        steer: float = float(actions[frame_idx][0])
        accel: float = float(actions[frame_idx][1])
        title_text.set_text(f"Frame {frame_idx} | Steer: {steer:.2f}, Accel: {accel:.2f}")
        return im, title_text

    # 4. Animation erzeugen
    anim: animation.FuncAnimation = animation.FuncAnimation(
        fig, update, frames=len(observations), interval=1000 / fps, blit=True
    )

    # Schließen der Figure, damit sie nicht doppelt als statisches Bild gerendert wird
    plt.close(fig)

    # Als HTML5-Video direkt in Colab einbetten
    return HTML(anim.to_jshtml())


def analyze_ood_start_conditions(env: gym.Env, num_episodes: int = 100) -> Figure:
    """
    Simuliert Episoden und analysiert die räumliche Verteilung des Autos,
    um zu beweisen, dass Out-of-Distribution (OOD) Zustände abgedeckt sind.
    """
    all_x: list[float] = []
    all_y: list[float] = []
    all_errors: list[float] = []

    # Für Plot 3 speichern wir die Distanz einer einzelnen "Rettungs-Episode"
    sample_trajectory_distances: list[float] = []

    print(f"Sammle Telemetriedaten für {num_episodes} Episoden...")

    for ep in range(num_episodes):
        env.reset()

        # Wir loggen nur die erste Episode komplett im Zeitverlauf
        log_trajectory = ep == 0

        for step in range(200):  # 200 Schritte reichen für die Analyse
            # Berechne Distanz zum Zentrum
            dist_to_center = np.sqrt(
                (env.car_x - env.track_center[0]) ** 2 + (env.car_y - env.track_center[1]) ** 2
            )
            error = abs(dist_to_center - env.track_radius)

            all_x.append(env.car_x)
            all_y.append(env.car_y)
            all_errors.append(error)

            if log_trajectory:
                sample_trajectory_distances.append(dist_to_center)

            # Teacher Action holen (epsilon=0.1 für leichten Drunken-Teacher-Effekt)
            action = get_teacher_action(
                env.car_x, env.car_y, env.car_theta, env.track_center, env.track_radius, epsilon=0.1
            )

            _, _, terminated, truncated, _ = env.step(action)

            if terminated or truncated:
                break

    # ==========================================
    # VISUALISIERUNG
    # ==========================================
    fig = Figure(figsize=(18, 5))
    axes = fig.subplots(1, 3)
    # fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Out-of-Distribution (OOD) Data Coverage Analysis", fontsize=16)

    # --- Plot 1: 2D Heatmap / Scatter ---
    # Zeigt, dass das Auto nicht nur auf einer perfekten Linie fährt
    axes[0].scatter(all_x, all_y, alpha=0.1, s=1, c="blue", label="Car Position")

    # Ideallinie einzeichnen
    theta = np.linspace(0, 2 * np.pi, 100)
    ideal_x = env.track_center[0] + env.track_radius * np.cos(theta)
    ideal_y = env.track_center[1] + env.track_radius * np.sin(theta)
    axes[0].plot(ideal_x, ideal_y, color="red", linewidth=2, label="Ideal Track (r=20)")

    axes[0].set_title("1. Spatial Distribution (2D)")
    axes[0].set_xlabel("X Coordinate")
    axes[0].set_ylabel("Y Coordinate")
    axes[0].axis("equal")
    axes[0].legend()

    # --- Plot 2: Histogramm des Fehlers ---
    # Zeigt die Varianz in den Daten. Ein perfekter (schlechter) Datensatz hätte hier
    # nur einen Balken bei 0.0. Wir wollen eine "Long Tail" sehen.
    axes[1].hist(all_errors, bins=50, color="purple", alpha=0.7)
    axes[1].axvline(x=15.0, color="red", linestyle="--", label="Terminate Threshold (15.0)")
    axes[1].set_title("2. Track Error Distribution")
    axes[1].set_xlabel("Distance from Ideal Line (Pixels)")
    axes[1].set_ylabel("Frequency (Frames)")
    axes[1].legend()

    # --- Plot 3: Recovery Trajectory (Time Series) ---
    # Beweist, dass der Teacher das Auto von Off-Track zurückführt
    axes[2].plot(sample_trajectory_distances, color="green", linewidth=2)
    axes[2].axhline(y=env.track_radius, color="red", linestyle="--", label="Ideal Track Radius")
    axes[2].set_title("3. Single Episode Recovery Trajectory")
    axes[2].set_xlabel("Time (Frames)")
    axes[2].set_ylabel("Distance to Track Center")
    axes[2].legend()

    fig.tight_layout()
    return fig
