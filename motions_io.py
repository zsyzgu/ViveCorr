import codecs
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dim = 72
known = 27
speed_limitation = 5

class Motion:
      def __init__(self):
            self.timestamp = []
            self.X_pos = []
            self.Y_pos = []
            self.X_start = []
            self.Y_start = []
            self.X_speed = []
            self.Y_speed = []

      def copy(self):
            motion = Motion()
            motion.timestamp = list(self.timestamp)
            motion.X_pos = list(self.X_pos)
            motion.Y_pos = list(self.Y_pos)
            motion.X_start = self.X_start
            motion.Y_start = self.Y_start
            motion.X_speed = list(self.X_speed)
            motion.Y_speed = list(self.Y_speed)
            return motion

      def settle(self):
            time_interval = np.diff(self.timestamp)
            self.X_speed = np.diff(self.X_pos, axis = 0)
            self.X_speed = [self.X_speed[i] / time_interval[i] for i in range(len(time_interval))]
            self.Y_speed = np.diff(self.Y_pos, axis = 0)
            self.Y_speed = [self.Y_speed[i] / time_interval[i] for i in range(len(time_interval))]

            self.timestamp = pd.rolling_mean(np.array(self.timestamp), 9)[8 : -1]
            self.X_pos = pd.rolling_mean(np.array(self.X_pos), 9)[8 : -1]
            self.Y_pos = pd.rolling_mean(np.array(self.Y_pos), 9)[8 : -1]
            self.X_speed = pd.rolling_mean(np.array(self.X_speed), 9)[8 :]
            self.Y_speed = pd.rolling_mean(np.array(self.Y_speed), 9)[8 :]
            self.X_start = self.X_pos[0]
            self.Y_start = self.Y_pos[0]
            self.X_pos = self.X_pos - self.X_start
            self.Y_pos = self.Y_pos - self.Y_start

            if (max([max(self.X_speed[i]) for i in range(len(self.X_speed))]) > speed_limitation):
                  return False
            if (max([max(self.Y_speed[i]) for i in range(len(self.Y_speed))]) > speed_limitation):
                  return False
            return True

      def add_start(self):
            self.X_pos = self.X_pos + self.X_start
            self.Y_pos = self.Y_pos + self.Y_start

      def output(self, file_path):
            self.add_start()
            output = codecs.open(file_path, 'w')
            
            for i in range(len(self.timestamp)):
                  info = str(self.timestamp[i])
                  info += ' ' + ' '.join([str(self.X_pos[i][j]) for j in range(len(self.X_pos[i]))])
                  info += ' ' + ' '.join([str(self.Y_pos[i][j]) for j in range(len(self.Y_pos[i]))])
                  output.write(info + '\n')

            output.close()

def load_motions(file_path):
      input = codecs.open(file_path, 'r')
      lines = input.readlines()
      input.close()

      raw_motion = {}
      for line in lines:
            tags = line.strip('\r\n').split(' ')

            motion_name = tags[0]
            motion_id = int(tags[1])
            timestamp = float(tags[2])

            if (raw_motion.has_key(motion_name) == False):
                  raw_motion[motion_name] = []

            if (timestamp == 0):
                  while (motion_id >= len(raw_motion[motion_name])):
                        raw_motion[motion_name].append(Motion())
                  motion = raw_motion[motion_name][motion_id]
                  motion.__init__()

            motion.timestamp.append(timestamp)
            motion.X_pos.append([float(tags[i]) for i in range(3, 3 + known)])
            motion.Y_pos.append([float(tags[i]) for i in range(3 + known, 3 + dim)])

      motions = {}
      for key in raw_motion:
            motions[key] = []
            for motion in raw_motion[key]:
                  if (motion.settle() == True):
                        motions[key].append(motion)

      return motions
