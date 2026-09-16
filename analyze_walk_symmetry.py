"""Native CPU gait diagnostics: path drift and phase-matched joint symmetry."""
from pathlib import Path
import argparse
import json
import numpy as np
import mujoco
import torch
from locomotion.native_policy import load_native_policy, bounded_policy_action
from locomotion.play_native import _joint_addresses, _control_ids, reset_robot, build_actor_observation
from locomotion.robot_profiles import get_robot_profile, make_native_model

MODES = {'forward': (.08, 0., 0.), 'left': (0., .06, 0.),
         'right': (0., -.06, 0.), 'curve_left': (.08, 0., .3),
         'curve_right': (.08, 0., -.3), 'stop': (0., 0., 0.)}
SWAP = [3, 4, 5, 0, 1, 2]
SIGN = np.array([-1, 1, 1, -1, 1, 1])


def rollout(policy, command, phase_offset, seconds):
    profile = get_robot_profile()
    model = make_native_model(profile)
    data = mujoco.MjData(model)
    qa, va = _joint_addresses(model, profile)
    controls = _control_ids(model, profile)
    reset_robot(model, data, qa, profile)
    action = np.zeros(6, dtype=np.float32)
    period = profile.gait_period_for_task(policy.task_id)
    reference = np.array([*data.qpos[:2], 0.])
    records = []
    for step in range(round(seconds / profile.control_dt)):
        obs = build_actor_observation(model, data, qa, va, action, np.asarray(command), profile, period)
        if phase_offset:
            obs[-2:] *= -1
        action = bounded_policy_action(policy, obs)
        data.ctrl[controls] = profile.action_target_from_normalized(action)
        for _ in range(profile.control_decimation):
            mujoco.mj_step(model, data)
        dt = profile.control_dt
        half = command[2] * dt / 2
        reference[:2] += dt * np.sinc(half / np.pi) * np.array([
            np.cos(reference[2]+half)*command[0]-np.sin(reference[2]+half)*command[1],
            np.sin(reference[2]+half)*command[0]+np.cos(reference[2]+half)*command[1]])
        reference[2] += command[2]*dt
        w,x,y,z = data.qpos[3:7]
        yaw = np.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z))
        records.append([data.time, *data.qpos[qa], *data.qpos[:2], yaw, *reference])
        if 1-2*(x*x+y*y) < .5:
            break
    a = np.asarray(records)
    yaw = np.unwrap(a[:, 9])
    valid = a[:, 0] >= 2.
    b = a[valid]
    metrics = {'survived_s': float(a[-1,0]), 'survived': len(a)==round(seconds/dt),
               'final_xy_error_m': (a[-1,7:9]-a[-1,10:12]).tolist(),
               'final_heading_error_deg': float(np.degrees(yaw[-1]-a[-1,12]))}
    if not len(b):
        return metrics, None
    metrics['path_rmse_m'] = float(np.sqrt(np.mean(np.sum((b[:,7:9]-b[:,10:12])**2,axis=1))))
    metrics['joint_mean_deg'] = np.degrees(b[:,1:7].mean(0)).tolist()
    metrics['joint_rom_deg'] = np.degrees(np.quantile(b[:,1:7],.95,axis=0)-np.quantile(b[:,1:7],.05,axis=0)).tolist()
    metrics['yaw_rate_radps'] = float(np.polyfit(b[:,0],yaw[valid],1)[0])
    # Fourier fit avoids rounding a 0.165 s half-cycle to 8 or 9 control ticks.
    phase = 2*np.pi*(b[:,0]/period+phase_offset)
    basis = np.stack([np.ones_like(phase)]+[f(k*phase) for k in range(1,5) for f in (np.sin,np.cos)],axis=1)
    coef = np.linalg.lstsq(basis,b[:,1:7],rcond=None)[0]
    grid = 2*np.pi*np.arange(100)/100
    gb = np.stack([np.ones_like(grid)]+[f(k*grid) for k in range(1,5) for f in (np.sin,np.cos)],axis=1)
    cycle = gb@coef
    metrics['cycle_fit_residual_deg'] = float(np.degrees(np.sqrt(np.mean((basis@coef-b[:,1:7])**2))))
    if command[1]==0 and command[2]==0:
        metrics['half_cycle_mirror_rmse_deg'] = np.degrees(np.sqrt(np.mean((cycle-np.roll(cycle[:,SWAP]*SIGN,50,axis=0))**2,axis=0))).tolist()
    return metrics, cycle


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('run',type=Path)
    p.add_argument('--iterations',type=int,nargs='+',required=True)
    p.add_argument('--seconds',type=float,default=12.)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    torch.set_num_threads(1)
    report={'run':str(args.run),'seconds':args.seconds,'environment':'nominal native MuJoCo, no DR; two initial clock phases', 'checkpoints':[]}
    for iteration in args.iterations:
        policy,_=load_native_policy(args.run/f'model_{iteration}.pt')
        entry={'iteration':iteration,'eligible':iteration>800,'phases':[]}
        for offset in (0.,.5):
            modes={}; cycles={}
            for name,command in MODES.items():
                modes[name],cycles[name]=rollout(policy,command,offset,args.seconds)
            for left,right in [('left','right'),('curve_left','curve_right')]:
                if cycles[left] is not None and cycles[right] is not None:
                    modes[left]['cross_command_mirror_rmse_deg']=np.degrees(np.sqrt(np.mean((cycles[left]-np.roll(cycles[right][:,SWAP]*SIGN,50,axis=0))**2,axis=0))).tolist()
            entry['phases'].append(modes)
        report['checkpoints'].append(entry)
        print(iteration, json.dumps(entry['phases'][0]),flush=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':
    main()
