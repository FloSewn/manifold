from manifold.racecar.data_generation import RaceEnv

# from manifold.racecar.vae import VAE


def test_environment_initialization():
    env = RaceEnv()
    # Beispiel: Prüfen, ob die Start-Dimensionen stimmen
    # assert env.state_space.shape == (64, 64, 3)
    assert env is not None
