import glob
import os
import sys
import random
import cv2
import numpy as np
import math
import time

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Conv2D, AveragePooling2D
from collections import deque
# from keras.applications.xception import Xception
# from keras.layers import Dense, GlobalAveragePooling2D
from keras.callbacks import TensorBoard
from keras.optimizers import Adam
from keras.models import Model, load_model
from threading import Thread
from tqdm import tqdm
import tensorflow as tf
import keras.backend.tensorflow_backend as backend


try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
import carla

IM_WIDTH = 320
IM_HEIGHT = 240
SHOW_SCREEN = False
SECOND_PER_EPISODE = 20
REPLAY_MEMORY_SIZE = 5_000
MIN_REPLAY_MEMORY_SIZE = 1_000
MINIBATCH_SIZE = 16
PREDICTION_BATCH_SIZE = 1
TRAINING_BATCH_SIZE = MINIBATCH_SIZE // 4
UPDATE_TARGET_EVERY = 5
MODEL_NAME = "new"
MEMORY_FRACTION = 0.4
MIN_REWARD = -200
EPISODES = 300  #the number of episode

DISCOUNT = 0.99
epsilon = 1
EPSILON_DECAY = 0.99
MIN_EPSILON = 0.2
AGGREGATE_STATS_EVERY = 10


class VehicleEvn:
    SHOW_CAMERA = SHOW_SCREEN  # Set whether to display the camera screen
    STEER_AMT = 1.0
    im_width = IM_WIDTH  # Setting the camera sensor image width
    im_height = IM_HEIGHT  # Setting the camera sensor image height
    front_camera = None
    actor_list = []
    collision_hist = []
    vehicles = []

    def __init__(self):
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(10.0)
        self.world = self.client.load_world('Town05_Opt')
        # Add a vehicle for training
        self.vehicle_main = self.world.get_blueprint_library().filter("vehicle.dodge_charger.police")[0]

        # Init NPC vehicles
        self.init_npc()

    # Add some NPC cars and pedestrians
    def init_npc(self):
        self.spawn_points = self.world.get_map().get_spawn_points()

        # Select some models from the blueprint library
        self.group = ['dodge', 'audi', 'tesla', 'mini', 'mustang', 'lincoln', 'prius', 'nissan', 'crown', 'impala',
                      'mercedes']
        self.blueprints = []
        for vehicle in self.world.get_blueprint_library().filter('*vehicle*'):
            if any(model in vehicle.id for model in self.group):
                self.blueprints.append(vehicle)

        # Set a max number of vehicles and prepare a list for those we spawn
        max_vehicles = 1
        max_vehicles = min([max_vehicles, len(self.spawn_points)])
        self.vehicles = []

        # Take a random sample of the spawn points and spawn some vehicles
        for i, spawn_point in enumerate(random.sample(self.spawn_points, max_vehicles)):
            temp = self.world.try_spawn_actor(random.choice(self.blueprints), spawn_point)
            if temp is not None:
                self.vehicles.append(temp)
        for vehicle in self.vehicles:
            vehicle.set_autopilot(True)

        # Set a max number of walkers and prepare a list for those we spawn
        max_walkers = 1
        max_walkers = min([max_walkers, len(self.spawn_points)])
        self.walkers = []
        self.controller = []
        # Spawn walkers
        for i, spawn_point in enumerate(random.sample(self.spawn_points, max_walkers)):
            pedestrian_bp = random.choice(self.world.get_blueprint_library().filter('*walker.pedestrian*'))
            temp2 = self.world.try_spawn_actor(pedestrian_bp, spawn_point)
            if temp2 is not None:
                # Spawn walker controllers
                controller_bp = self.world.get_blueprint_library().find('controller.ai.walker')
                ai_controller = self.world.spawn_actor(controller_bp, carla.Transform(), temp2)
                self.controller.append(ai_controller)
                self.walkers.append(temp2)
        for people in self.controller:
            # Set up pedestrian navigation
            people.start()
            people.go_to_location(self.world.get_random_location_from_navigation())

    def set(self):
        self.actor_list = []
        self.collision_hist = []
        # Add the training vehicle
        self.spawn_points = self.world.get_map().get_spawn_points()
        self.vehicle = self.world.spawn_actor(self.vehicle_main, random.choice(self.spawn_points))
        self.actor_list.append(self.vehicle)

        # Add spectator
        self.spectator = self.world.get_spectator()
        self.spectator_vehicle_offset = carla.Location(x=-3, z=3)  # offset behind the vehicle
        vehicle_transform = self.vehicle.get_transform()
        new_transform = carla.Transform(vehicle_transform.location + self.spectator_vehicle_offset)
        self.spectator.set_transform(new_transform)

        # Defining the sensor - RGB camera
        self.rgb_cam = self.world.get_blueprint_library().find('sensor.camera.rgb')
        self.rgb_cam.set_attribute("image_size_x", f"{self.im_width}")
        self.rgb_cam.set_attribute("image_size_y", f"{self.im_height}")
        self.rgb_cam.set_attribute("fov", f"110")  # Setting up the front camera

        transform = carla.Transform(carla.Location(x=2.5, z=0.7))
        self.sensor = self.world.spawn_actor(self.rgb_cam, transform, attach_to=self.vehicle)
        self.actor_list.append(self.sensor)
        self.sensor.listen(lambda data: self.image_data(data))

        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
        time.sleep(4)

        # collision sensor
        collsensor = self.world.get_blueprint_library().find('sensor.other.collision')
        self.collsensor = self.world.spawn_actor(collsensor, transform, attach_to=self.vehicle)
        self.actor_list.append(self.collsensor)
        self.collsensor.listen(lambda event: self.collision_data(event))

        while self.front_camera is None:
            time.sleep(0.01)
        self.episode_start = time.time()
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))

        return self.front_camera

     # Callback functions
    def collision_data(self, event):
        self.collision_hist.append(event)

    # The function converts an image from a 1D array form to a 3D form
    def image_data(self, image):
        i = np.array(image.raw_data)
        i2 = i.reshape((self.im_height, self.im_width, 4))
        i3 = i2[:, :, : 3]
        if self.SHOW_CAMERA:
            cv2.imshow("", i3)
            cv2.waitKey(1)
        self.front_camera = i3

    def step(self, action):
        # Defining the movement of the car
        if action == 0:
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=-1 * self.STEER_AMT))  # left
        elif action == 1:
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=0))
        elif action == 2:
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=1 * self.STEER_AMT))   # right
        elif action == 3:
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.5, steer=0))  # slow down
        elif action == 4:
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=0.5 * self.STEER_AMT))  # slight right turn
        elif action == 5:
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=-0.5 * self.STEER_AMT))  # slight left turn

        # Determining whether the brakes are needed
        if abs(self.vehicle.get_control().steer) > 0.5:
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.8, steer=self.vehicle.get_control().steer, brake=0.4))
        else:
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=self.vehicle.get_control().steer, brake=0.0))


        v = self.vehicle.get_velocity()
        kmh = int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)) # Converting the speed of a vehicle from vector to scalar form
        if len(self.collision_hist) != 0:
            done = True
            reward = -200

        elif kmh < 50:
            done = False
            reward = -1
        else:
            done = False
            reward = 1

        if self.episode_start + SECOND_PER_EPISODE < time.time():
            done = True

        return self.front_camera, reward, done, None

# Own Tensorboard class
class ModifiedTensorBoard(TensorBoard):

    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.FileWriter(self.log_dir)

    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        pass

    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Overrided
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    # Overrided, so won't close writer
    def on_train_end(self, _):
        pass

    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        self._write_logs(stats, self.step)


class DQN:
    def __init__(self):
        self.model = load_model('C:/Users/qsy/PycharmProjects/Carla/another/new.model')  # create an evaluation network model
        self.target_model = self.create_model()  # create a target network model
        self.target_model.set_weights(self.model.get_weights())  # set target network's weights using evaluation network's weights

        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)  # double-ended queue
        self.tensorboard = ModifiedTensorBoard(log_dir=f"logs/{MODEL_NAME}-{int(time.time())}")

        self.graph = tf.get_default_graph()
        self.terminate = False
        self.training_initialized = False
        self.last_logged_episode = 0
        self.target_counter = 0

   # This code defines a neural network model. It contains 3 convolutional layers, each containing 64 filters, each of
    # size 3x3. A ReLU (Rectified Linear Unit) activation function is added after each convolutional layer and an average pooling layer of
    # size 5x5 with a step size of 3x3.

    def create_model(self):
        model = Sequential()
        model.add(Conv2D(64, (3, 3), input_shape=(IM_HEIGHT, IM_WIDTH, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='same'))

        model.add(Conv2D(64, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='same'))

        model.add(Conv2D(64, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='same'))

        model.add(Flatten())
        model.add(Dense(512))  # A fully connected layer with 512 neurons and a ReLU activation function is then added
        model.add(Activation('relu'))
        # a fully connected layer with 3 neurons and a linear activation function is added to output the Q value.
        model.add(Dense(3, activation='linear'))
        # Uses the mean square error (MSE) as the loss function and is optimised using the Adam optimiser.
        model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=['accuracy'])

        return model

    def update_replay_memory(self, transition):
        # transition = (current_state, action, reward, new_state, done)
        self.replay_memory.append(transition)


# Use deep RL algorithms to optimise the weights of the neural network to better predict the Q
# value of each action, enabling the model to learn and make optimal action choices in different environments

    def train(self):
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:  # Check if the memory meets the minimum size requirement, if not then do not train
            return
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)  # Random sampling of a batch of data from a memory

        # Use the current(evaluation) model and the target model to predict the Q value of the current state and the next state respectively
        # The current state is converted to a numpy array and normalised. Then, the current state is predicted using--
        # --the current model (self.model) and the predicted Q values are stored in the current_qs_list
        current_states = np.array([transition[0] for transition in minibatch]) / 255
        with self.graph.as_default():
            current_qs_list = self.model.predict(current_states, PREDICTION_BATCH_SIZE)
        future_states = np.array([transition[3] for transition in minibatch]) / 255
        with self.graph.as_default():
            future_qs_list = self.target_model.predict(future_states, PREDICTION_BATCH_SIZE)

        Status = []
        q_value = []
        # Q-value update formula
        for index, (current_state, action, reward, new_state, done) in enumerate(minibatch):
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q  # Calculate new Q value
            else:
                new_q = reward

            current_qs = current_qs_list[index]
            current_qs[action] = new_q   # update Q value of the action of the current state
            Status.append(current_state)
            q_value.append(current_qs)

        # Determine if the current training information needs to be recorded by checking if the current training step count is
        # greater than the last recorded step count
        log_this_step = False
        if self.tensorboard.step > self.last_logged_episode:
            log_this_step = True
            self.last_logged_episode = self.tensorboard.step
        # Train the model
        with self.graph.as_default():
                # input/output/the number of samples used in each training session/not output the log of the training process/not disrupt the order of the training data
            self.model.fit(np.array(Status)/255, np.array(q_value), batch_size=TRAINING_BATCH_SIZE, verbose=0, shuffle=False, callbacks=[self.tensorboard] if log_this_step else None)

        if log_this_step:
            self.target_counter += 1

        # Ensure that the target model is updated regularly to track improvements to the current model
        if self.target_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_counter = 0

# Continuous training of DQN models
    def train_in_loop(self):
        # Generate a random input x and output y for initialising the model
        x = np.random.uniform(size=(1, IM_HEIGHT, IM_WIDTH, 3)).astype(np.float32)
        y = np.random.uniform(size=(1, 3)).astype(np.float32)
        with self.graph.as_default():
            self.model.fit(x, y, verbose=False, batch_size=1)

        self.training_initialized = True

        while True:
            if self.terminate:
                return
            self.train()
            time.sleep(0.01)

# Get the Q value of the current state (state)
    def get_qs(self, state):
        return self.model.predict(np.array(state).reshape(-1, *state.shape)/255)[0]


if __name__ == '__main__':
    # Percentage of memory allocated to GPU
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=MEMORY_FRACTION)
    backend.set_session(tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)))

    FPS = 60
    ep_rewards = [-200]
    random.seed(1)
    np.random.seed(1)
    tf.set_random_seed(1)
    if not os.path.isdir('another'):
        os.makedirs('another')
    agent = DQN()
    env = VehicleEvn()

    trainer_thread = Thread(target=agent.train_in_loop, daemon=True) # The training loop for the model is started in a separate thread
    trainer_thread.start()
    while not agent.training_initialized:
        time.sleep(0.01)

    agent.get_qs(np.ones((env.im_height, env.im_width, 3)))

    # tqdm--Show progress bar--The program starts iterating over each episode
    for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit="episodes"):
        env.collision_hist = []  # Clear the collision history before each episode starts
        agent.tensorboard.step = episode  # Set the number of steps in the tensorboard to the current number of episodes
        episode_reward = 0
        step = 1
        current_state = env.set()
        done = False
        episode_start = time.time()

        while True:
# Take an action in the current state, if the random number is greater than epsilon, take the optimal action predicted
# by the model, otherwise choose the action randomly
            if np.random.random() > epsilon:
                action = np.argmax(agent.get_qs(current_state))
            else:
                action = np.random.randint(0, 3)
                time.sleep(1/FPS)

            new_state, reward, done, _ = env.step(action)
            episode_reward += reward
            agent.update_replay_memory((current_state, action, reward, new_state, done)) # Store the current experience in the experience replay
            current_state = new_state
            step += 1

            if done:
                break

        for actor in env.actor_list:
            actor.destroy()


        # Append episode reward to a list and log stats(every given number of episodes)
        ep_rewards.append(episode_reward)
        if not episode % AGGREGATE_STATS_EVERY or episode == 1:
            average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:]) / len(ep_rewards[-AGGREGATE_STATS_EVERY:])
            min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
            max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
            agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward, epsilon=epsilon)

            # Save model, but only when min reward is greater or equal a set value
            if min_reward >= MIN_REWARD:
                agent.model.save(f'another/{MODEL_NAME}.model')

        # Decay epsilon makes the intelligences gradually explore less and less during training and gradually move to better actions
        if epsilon > MIN_EPSILON:
            epsilon *= EPSILON_DECAY
            epsilon = max(MIN_EPSILON, epsilon)

    # Set termination
    agent.terminate = True
    trainer_thread.join()
    agent.model.save(f'another/{MODEL_NAME}.model')


