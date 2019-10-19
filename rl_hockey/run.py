from PIL import Image, ImageTk
from rl_hockey.physics import rigid_body_physics
import torch
import time
import numpy as np


def run(memory, world, numSteps=1500, canvas=None, root=None, draw_step=1, draw_scale=0.5, pause_time=0):
    world.reset()
    num_cpu = world.get_num_cpu()
    frame_skip = 4
    timestep = 1 / 30

    state = world.get_state()
    score = world.get_scores()

    for i in range(numSteps):
        start = time.time()
        if i % frame_skip == 0 and i > 0: # Update memory
            prev_score = score
            score = world.get_scores()
            done = world.terminate_run()
            prev_state = state.copy()
            state = world.get_state()
            all_actions = world.get_last_action()

            for j in range(num_cpu):
                reward = torch.tensor([(score[j] - prev_score[j])], dtype=torch.float)
                action = torch.tensor([all_actions[j]])
                memory[j].push(prev_state[j], action, reward, state[j], done=done)

            if done:  # Check if we should terminate, then do so
                break


        if i % frame_skip == 0:  # On first frame of block apply control
            world.apply_control(state=state, frame_skip=False)  # Update state of world
        else:
            world.apply_control(state=state, frame_skip=True)  # Update state of world

        rigid_body_physics(world, timestep)  # Move pieces
        world.update_score()                       # Update world, including score

        if canvas is not None and i % draw_step == 0:

            #arr = world.get_state()
            arr = world.draw_world(high_scale=draw_scale)
            # arr = state[0,1,:,:] # Diff state
            arr = arr.transpose([1, 0, 2])
            img = ImageTk.PhotoImage(image=Image.fromarray(arr.astype(np.uint8), mode='RGB'))
            canvas.create_image(0, 0, anchor='nw', image=img)
            root.update()
            end = time.time()
            # Time taken by controller can vary, calcualte time to pause based on how much time has elapsed since last frame
            time.sleep(np.max([0.00, pause_time - (end - start)]))

    return memory