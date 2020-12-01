"""robot`Physics` implementation and helper classes.
"""
import socket

import time
import numpy as np
import time
import json
from skimage.transform import resize


class RobotClient():
    def __init__(self, robot_ip="127.0.0.1", port=9030):
        self.robot_ip = robot_ip
        self.port = port
        self.connected = False
        self.startseq = '<|'
        self.endseq = '|>'
        self.midseq = '**'

    def connect(self):
        while not self.connected:
            print("attempting to connect with robot at {}".format(
                self.robot_ip))
            self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.tcp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR,
                                       1)
            self.tcp_socket.settimeout(100)
            # connect to computer
            self.tcp_socket.connect((self.robot_ip, self.port))
            print('connected')
            self.connected = True
            if not self.connected:
                time.sleep(1)

    def decode_state(self, robot_response):
        #print('decoding', robot_response)
        ackmsg, resp = robot_response.split('**')
        # successful msg has ACKSTEP
        assert ackmsg[:5] == '<|ACK'
        # make sure we got msg end
        assert resp[-2:] == '|>'
        vals = [x.split(': ')[1] for x in resp[:-2].split('\n')]
        # deal with each data type
        success = bool(vals[0])
        robot_msg = eval(vals[1])
        # not populated
        joint_names = vals[2]
        # num states seen in this step
        self.n_state_updates = int(vals[3])
        timediff = json.loads(vals[4])[-1]
        joint_position = json.loads(vals[5])
        joint_velocity = json.loads(vals[6])
        joint_effort = json.loads(vals[7])
        tool_pose = json.loads(vals[8])
        #print('returning from decode state')
        return timediff, joint_position, joint_velocity, joint_effort, tool_pose

    def send(self, cmd, msg='XX'):
        packet = self.startseq + cmd + self.midseq + msg + self.endseq
        self.tcp_socket.sendall(packet.encode())
        # TODO - should prob handle larger packets
        self.tcp_socket.settimeout(100)
        rx = self.tcp_socket.recv(2048).decode()
        return rx

    def render(self):
        packet = self.startseq + "RENDER" + self.midseq + "XX" + self.endseq
        self.tcp_socket.settimeout(100)
        self.tcp_socket.sendall(packet.encode())
        self.tcp_socket.settimeout(100)
        # TODO - should prob handle larger packets
        rxl = []
        rxing = True
        cnt = 0
        end = self.endseq.encode()
        while rxing:
            rx = self.tcp_socket.recv(2048)
            rxl.append(rx)
            cnt += 1
            # byte representation of endseq
            if rx[-2:] == end:
                rxing = False
        allrx = b''.join(rxl)[2:-2]
        # height, width
        img = np.frombuffer(allrx, dtype=np.uint8).reshape(480, 640, 3)
        # right now cam is rotated
        #img = (img* 255).astype(np.uint8)
        #image_enc = vals[9]
        #image_height = int(vals[10])
        #image_width = int(vals[11])
        #image_data = vals[12]
        #image_dict = {'enc':image_enc,
        #              'height':image_height,
        #              'width':image_width,
        #              'data':image_data}
        return img

    def home(self):
        return self.send('HOME')

    def reset(self):
        print('Robot Client sending reset')
        return self.decode_state(self.send('RESET'))

    def get_state(self):
        return self.decode_state(self.send('GET_STATE'))

    def initialize(self, minx, maxx, miny, maxy, minz, maxz):
        data = '{},{},{},{},{},{}'.format(minx, maxx, miny, maxy, minz, maxz)
        return self.decode_state(self.send('INIT', data))

    def step(self, command_type, relative, unit, data):
        assert (command_type in ['VEL', 'ANGLE', 'TOOL'])
        datastr = ','.join(['%.4f' % x for x in data])
        data = '{},{},{},{}'.format(command_type, 0, unit, datastr)
        #print("STEP", data)
        return self.decode_state(self.send('STEP', data))

    def end(self):
        self.send('END')
        print('disconnected from {}'.format(self.robot_ip))
        self.tcp_socket.close()
        self.connected = False


class JacoPhysics():
  """Encapsulates a robot interface.

  # Apply controls and advance the simulation state.
  physics.set_control(np.random.random_sample(size=N_ACTUATORS))
  physics.step()

  # Render a camera defined in the NumPy array.
  rgb = physics.render(height=240, width=320, id=0)
  """
  def __init__(self,
                robot_name='j2s7s300',
                robot_server_ip='127.0.0.1',
                robot_server_port=9030,
                fence={
                    'x': [-.5, .5],
                    'y': [-.5, .3],
                    'z': [0.1, 1.2]
                },
                control_type='position'):
      self.type = 'robot'
      # only compatible with j2
      robot_model = robot_name[:2]
      assert robot_model == 'j2'
      # only tested with 7dof, though 6dof should work with tweaks
      self.n_major_actuators = int(robot_name[3:4])
      assert self.n_major_actuators == 7
      # only tested with s3 hand
      hand_type = robot_name[4:6]
      assert hand_type == 's3'
      if hand_type == 's3':
          self.n_hand_actuators = 6
      self.n_actuators = self.n_major_actuators + self.n_hand_actuators

      self.fence = fence
      self.control_type = 'position'
      self.n_actuators = 13
      self.data = np.zeros(self.n_actuators)
      self.experiment_timestep = 0
      self.robot_server_ip = robot_server_ip
      self.robot_server_port = robot_server_port
      # todo - require confirmation of fence?
      self.robot_client = RobotClient(robot_ip=self.robot_server_ip,
                                      port=self.robot_server_port)
      self.robot_client.connect()
      resp = self.robot_client.initialize(min(self.fence['x']),
                                          max(self.fence['x']),
                                          min(self.fence['y']),
                                          max(self.fence['y']),
                                          min(self.fence['z']),
                                          max(self.fence['z']))
      self.handle_state(resp)
      self.image_dict = {
          'enc': 'none',
          'width': 0,
          'height': 0,
          'data': 'none'
      }

  def step(self, control):
    """Advances physics with up-to-date position and velocity dependent fields.
    """
    # TODO - only step once for real robot
    self.handle_state(
        self.robot_client.step(command_type='ANGLE',
                                relative=False,
                                unit='rad',
                                data=control))
    return self.get_state()

  def render(self,
              height=640,
              width=480,
              camera_id=-1,
              overlays=(),
              depth=False,
              segmentation=False,
              scene_option=None):
    """
    Args:
      height: Viewport height (number of pixels). Optional, defaults to 240.
      width: Viewport width (number of pixels). Optional, defaults to 320.
      camera_id: Optional camera name or index. Defaults to -1, the free
        camera, which is always defined. A nonnegative integer or string
        corresponds to a fixed camera, which must be defined in the model XML.
        If `camera_id` is a string then the camera must also be named.
      overlays: An optional sequence of `TextOverlay` instances to draw. Only
        supported if `depth` is False.
      depth: If `True`, this method returns a NumPy float array of depth values
        (in meters). Defaults to `False`, which results in an RGB image.
      segmentation: If `True`, this method returns a 2-channel NumPy int32 array
        of label values where the pixels of each object are labeled with the
        pair (mjModel ID, mjtObj enum object type). Background pixels are
        labeled (-1, -1). Defaults to `False`, which returns an RGB image.
      scene_option: An optional `wrapper.MjvOption` instance that can be used to
        render the scene with custom visualization options. If None then the
        default options will be used.

    Returns:
      The rendered RGB, depth or segmentation image.
    """
    img = self.robot_client.render()
    img = resize(img, (width, height))
    # skcit op changes the type, revert it back
    img = (img* 255).astype(np.uint8)
    return img

  def get_state(self):
    """Returns the physics state.
    Returns:
      NumPy array containing full physics simulation state.
    """
    return np.concatenate(self._physics_state_items())

  def _physics_state_items(self):
    """Returns list of arrays making up internal physics simulation state.

    The physics state consists of the state variables, their derivatives and
    actuation activations.

    Returns:
      List of NumPy arrays containing full physics simulation state.
    """
    return [
        self.actuator_position, self.actuator_velocity,
        self.actuator_effort
    ]

  def handle_state(self, state_tuple):
    timediff, joint_position, joint_velocity, joint_effort, tool_pose = state_tuple
    self.timediff = timediff
    self.actuator_position = np.array(joint_position)
    self.actuator_velocity = np.array(joint_velocity)
    self.actuator_effort = np.array(joint_effort)
    self.tool_pose = np.array(tool_pose)

  def reset(self):
    """Resets internal variables of the physics simulation."""
    print('JacoPhysics reset')
    self.n_steps = 0
    self.experiment_timestep = 0
    self.handle_state(self.robot_client.reset())
    return self.get_state()

  def after_reset(self):
    """Runs after resetting internal variables of the physics simulation."""
    # Disable actuation since we don't yet have meaningful control inputs.
    #self.robot_client.end()
    pass

  def __getstate__(self):
    return self.data  # All state is assumed to reside within `self.data`.

  def _physics_state_items(self):
    """Returns list of arrays making up internal physics simulation state.

    The physics state consists of the state variables, their derivatives and
    actuation activations.

    Returns:
      List of NumPy arrays containing full physics simulation state.
    """
    return [
        self.actuator_position, self.actuator_velocity,
        self.actuator_effort
    ]

  # def control(self):
  #   """Returns a copy of the control signals for the actuators."""
  #   return self.control_action

  def state(self):
      """Returns the full physics state. Alias for `get_physics_state`."""
      return np.concatenate(self._physics_state_items())

  # def position(self):
  #     """Returns a copy of the generalized positions (system configuration)."""
  #     return self.actuatory_position

  # def velocity(self):
  #     """Returns a copy of the generalized velocities."""
  #     return self.actuator_velocity()

  def timestep(self):
    """Returns the timestep."""
    # TODO - set this to .001 to match xml file for mujoco simulation - this is a hack for position control,
    # but won't work for velocity/torque control. Ros runs at ~52 Hz.
    return .001

  def time(self):
    """Returns episode time in seconds."""
    return self.experiment_timestep
