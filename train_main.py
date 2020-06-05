import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import sys
import time
import pygame
import pydub
import pprint
from Env import NaiveBandori
import argparse
import numpy as np
import random

import torch
from torch.utils.tensorboard import SummaryWriter
from NaiveVectorEnv import MyVectorEnv as VectorEnv
# from tianshou.env import VectorEnv
from tianshou.policy import PPOPolicy
from tianshou.trainer import onpolicy_trainer
from tianshou.data import Collector, ReplayBuffer
from PreliminaryBandoriNet import PreBandoriPPO

parser = argparse.ArgumentParser()


def get_args():
    # parser.add_argument('--task', type=str, default='CartPole-v0')
    parser.add_argument('-seed', type=int, default=None)
    parser.add_argument('--buffer-size', type=int, default=20000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--step-per-epoch', type=int, default=2000)
    parser.add_argument('--collect-per-step', type=int, default=20)
    parser.add_argument('--repeat-per-collect', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--layer-num', type=int, default=1)
    parser.add_argument('--training-num', type=int, default=65)
    parser.add_argument('--test-num', type=int, default=100)
    parser.add_argument('--logdir', type=str, default='log')
    parser.add_argument('--render', type=float, default=0.)
    parser.add_argument(
        '--device', type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu')
    # ppo special
    parser.add_argument('--vf-coef', type=float, default=0.5)
    parser.add_argument('--ent-coef', type=float, default=0.0)
    parser.add_argument('--eps-clip', type=float, default=0.2)
    parser.add_argument('--max-grad-norm', type=float, default=0.5)
    parser.add_argument('--gae-lambda', type=float, default=0.8)
    parser.add_argument('--rew-norm', type=bool, default=True)
    parser.add_argument('--dual-clip', type=float, default=None)
    parser.add_argument('--value-clip', type=bool, default=True)
    return parser.parse_known_args()[0]


if __name__ == '__main__':
    parser.add_argument("-t", "--thread_num", dest="thread_num", help="torch thread num", type=int, default=6)
    parser.add_argument("-mode", "--running_mode", dest="mode", help="input 'train' if u wanna train", type=str, default='train')
    parser.add_argument("-a, ", "--audio", dest="audio_state", help="train with audio set 1, else 0", type=int, default=1)
    parser.add_argument("-test_times, ", "--test_times", dest="test_times", help="test times", type=int, default=1)
    parser.add_argument("-no_load, ", "--no_load", dest="no_load", help="no load model", type=int, default=1)
    args = get_args()
    # args.seed = None
    torch.set_num_threads(args.thread_num)
    audio_state = True if args.audio_state != 0 else False
    # env/agent global parameters
    last_best = './log/0527-003808-A-/ppo/policy.pth'
    height = 90  # width must be int!!
    interval = 3
    # os parameters
    if sys.platform.startswith('win'):
        pydub.AudioSegment.ffmpeg = './ffmpeg.exe'
        pydub.AudioSegment.ffprobe = './ffprobe.exe'
    if args.seed is not None:  # seed
        print(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    # model
    actor = PreBandoriPPO(
        is_actor=True,
        input_channel=3,
        interval=interval,
        audio_state=audio_state,
        audio_input_size=129*37,
        audio_net_layer=2,
        audio_mid_size=256,
        audio_output_size=256,
        device=args.device
    ).to(args.device)
    critic = PreBandoriPPO(
        is_actor=False,
        input_channel=3,
        interval=interval,
        audio_state=audio_state,
        audio_input_size=129 * 37,
        audio_net_layer=2,
        audio_mid_size=256,
        audio_output_size=256,
        device=args.device
    ).to(args.device)
    # orthogonal initialization
    for m in list(actor.modules()) + list(critic.modules()):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.orthogonal_(m.weight)
            torch.nn.init.zeros_(m.bias)
    optim = torch.optim.Adam(list(
        actor.parameters()) + list(critic.parameters()), lr=args.lr)
    dist = torch.distributions.Categorical
    policy = PPOPolicy(
        actor, critic, optim, dist, args.gamma,
        max_grad_norm=args.max_grad_norm,
        eps_clip=args.eps_clip,
        vf_coef=args.vf_coef,
        ent_coef=args.ent_coef,
        action_range=None,
        gae_lambda=args.gae_lambda,
        reward_normalization=args.rew_norm,
        dual_clip=args.dual_clip,
        value_clip=args.value_clip)
    if not args.no_load:
        if os.path.exists(last_best):
            policy.load_state_dict(torch.load(last_best))  # .to(args.device)

    if args.mode == 'train':
        HEADLESS = 1
        if HEADLESS > 0:
            os.environ['SDL_AUDIODRIVER'] = 'dummy'
            # os.environ['SDL_DISKAUDIOFILE'] = '/root/audio' -> for disk driver
            # if a server has a GPU, video dummy is ok, the same as my PC.
            os.environ["SDL_VIDEODRIVER"] = "dummy"
            pass
        torch.cuda.empty_cache()
        # pygame initialisation
        pygame.mixer.pre_init(frequency=48000)
        pygame.mixer.init(channels=1)
        pygame.init()
        pygame.display.set_caption('MultiMediaRL')  # title

        # if args.seed is not None:
        #     train_envs.seed(args.seed)
        #     test_envs.seed(args.seed)
        train_envs = VectorEnv([
            NaiveBandori(
                height=height,
                noteSpeed=9.0,
                interval=interval,
                audio_state=audio_state,
                real_music=not True,
                seed=args.seed) for _ in range(args.training_num)])
        # test_envs = gym.make(args.task)
        test_envs = VectorEnv([
            NaiveBandori(
                height=height,
                noteSpeed=9.0,
                interval=interval,
                audio_state=audio_state,
                real_music=not True,
                seed=args.seed) for _ in range(args.test_num)])

        # collector
        train_collector = Collector(
            policy, train_envs, ReplayBuffer(args.buffer_size))
        test_collector = Collector(policy, test_envs)
        # log
        if args.seed is not None:
            seed_mark = str(args.seed)
        else:
            seed_mark = ''
        log_path = os.path.join(args.logdir, time.strftime('%m%d-%H%M%S')+('-A' if audio_state else '')+'-'+seed_mark,  'ppo')
        writer = SummaryWriter(log_path)

        def save_fn(policy):
            torch.save(policy.state_dict(), os.path.join(log_path, 'policy.pth'))

        def stop_fn(x):
            return x >= 289  # env.spec.reward_threshold
        # trainer
        result = onpolicy_trainer(
            policy, train_collector, test_collector, args.epoch,
            args.step_per_epoch, args.collect_per_step, args.repeat_per_collect,
            args.test_num, args.batch_size, stop_fn=stop_fn, save_fn=save_fn,
            writer=writer)
        assert stop_fn(result['best_reward'])
        train_collector.close()
        test_collector.close()
        pprint.pprint(result)
        pygame.quit()
    # Let's watch its performance!
    if sys.platform.startswith('win') and audio_state:
        os.environ['SDL_AUDIODRIVER'] = 'dsound'
        os.environ["SDL_VIDEODRIVER"] = "directx"
        for _ in range(args.test_times):
            pygame.mixer.pre_init(frequency=48000)
            pygame.mixer.init(channels=1)
            pygame.init()
            pygame.display.set_caption('MultiMediaRL')  # title
            e = NaiveBandori(
                height=height,
                noteSpeed=9.0,
                interval=interval,
                audio_state=audio_state,
                real_music=True,
                seed=args.seed)
            r = e.show_for_act(policy.actor)
            print(r)
            pygame.mixer.quit()
            pygame.quit()
            os.remove('a.wav')
