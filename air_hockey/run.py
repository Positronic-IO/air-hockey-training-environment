import logging
import time

import numpy as np
import torch

from air_hockey.connect import RedisConnection
from air_hockey.physics import rigid_body_physics
from PIL import Image, ImageTk

redis = RedisConnection()

# Initiate Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def run(memory, world, numSteps=1500, canvas=None, root=None, draw_step=1, draw_scale=0.5, pause_time=0):
    # world.reset(total=True)
    world.reset()
    num_cpu = world.get_num_cpu()
    frame_skip = 4
    timestep = 1 / 30

    state = world.get_state()
    score = world.get_scores()
    # done = False
    # i = 0
    # prev_left_score = prev_right_score = 0
    # while True:
    for i in range(numSteps):
        start = time.time()
        if i % frame_skip == 0 and i > 0:  # Update memory
            prev_score = score
            score = world.get_scores()
            done = world.terminate_run()
            prev_state = state.copy()
            state = world.get_state()
            all_actions = world.get_last_action()

            reward = torch.tensor([(score[0] - prev_score[0])], dtype=torch.float)
            action = torch.tensor([all_actions[0]])
            memory.push(prev_state[0], action, reward, state[0], done=done)
            
            if done: 
                break

        if i % frame_skip == 0:  # On first frame of block apply control
            world.apply_control(state=state, frame_skip=False)  # Update state of world
        else:
            world.apply_control(state=state, frame_skip=True)  # Update state of world

        rigid_body_physics(world, timestep)  # Move pieces
        world.update_score()  # Update world, including score

        # Draw screen
        if canvas is not None and i % draw_step == 0:
            
            puck_location = tuple(world.puck.x)
            left_location = tuple(world.player_list[0].obj.x)
            right_location = tuple(world.player_list[1].obj.x)
            redis.post(
                {
                    "components": {
                        "puck": {"location": puck_location},
                        "robot": {"location": left_location},
                        "opponent": {"location": right_location},
                    }
                }
            )
    
            # Alert the positions and scores are different
            redis.publish("position-update")
            redis.publish("score-update")



            # arr = world.get_state()
            arr = world.draw_world(high_scale=draw_scale)
            # arr = state[0,1,:,:] # Diff state
            arr = arr.transpose([1, 0, 2])
            img = ImageTk.PhotoImage(image=Image.fromarray(arr.astype(np.uint8), mode="RGB"))
            canvas.create_image(0, 0, anchor="nw", image=img)
            root.update()
            end = time.time()
            # Time taken by controller can vary, calcualte time to pause based on how much time has elapsed since last frame
            time.sleep(np.max([0.00, pause_time - (end - start)]))

        # if prev_left_score < world.left_score:
        #     redis.post({"scores": {"robot_score": world.left_score, "opponent_score": world.right_score}})
        #     prev_left_score = world.left_score

        # if prev_right_score < world.right_score:
        #     redis.post({"scores": {"robot_score": world.left_score, "opponent_score": world.right_score}})
        #     prev_right_score = world.right_score

        # if prev_right_score == 10 or prev_left_score == 0:  # Check if we should terminate, then do so
        #     print("New game")
        #     break

        # i += 1
    return memory
