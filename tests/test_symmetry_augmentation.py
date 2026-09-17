import torch
from tensordict import TensorDict

from locomotion.symmetry import leggybot_left_right_augmentation, mirror_leggybot_actions


def test_leggybot_symmetry_augmentation_mirrors_actor_critic_and_actions() -> None:
    actor = torch.arange(58, dtype=torch.float32).reshape(2, 29)
    critic = torch.arange(88, dtype=torch.float32).reshape(2, 44)
    actions = torch.arange(12, dtype=torch.float32).reshape(2, 6)
    obs = TensorDict({"actor": actor, "critic": critic}, batch_size=[2])

    augmented_obs, augmented_actions = leggybot_left_right_augmentation(
        None, obs, actions
    )

    assert augmented_obs.batch_size == torch.Size([4])
    assert augmented_actions.shape == (4, 6)
    torch.testing.assert_close(augmented_obs["actor"][:2], actor)
    torch.testing.assert_close(augmented_obs["critic"][:2], critic)
    torch.testing.assert_close(augmented_obs["actor"][2:, 27:29], -actor[:, 27:29])
    torch.testing.assert_close(augmented_obs["critic"][2:, 42:44], -critic[:, 42:44])
    torch.testing.assert_close(augmented_obs["actor"][2:, 6], -actor[:, 9])
    torch.testing.assert_close(augmented_obs["actor"][2:, 7], actor[:, 10])
    torch.testing.assert_close(augmented_obs["critic"][2:, 0], critic[:, 0])
    torch.testing.assert_close(augmented_obs["critic"][2:, 1], -critic[:, 1])
    torch.testing.assert_close(augmented_obs["critic"][2:, 30], critic[:, 31])
    torch.testing.assert_close(augmented_obs["critic"][2:, 37], -critic[:, 40])
    torch.testing.assert_close(augmented_actions[2:], mirror_leggybot_actions(actions))


def test_leggybot_symmetry_transform_rejects_an_abi_change() -> None:
    obs = TensorDict(
        {"actor": torch.zeros(1, 28), "critic": torch.zeros(1, 44)}, batch_size=[1]
    )
    try:
        leggybot_left_right_augmentation(None, obs, torch.zeros(1, 6))
    except ValueError as error:
        assert "29 actor values" in str(error)
    else:
        raise AssertionError("the fixed actor ABI must be validated")
