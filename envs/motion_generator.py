from posixpath import join
import numpy as np
import json
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from IPython import embed

class MotionGenerator():
    '''
    this class is used to generate motions for the character: batch version
    '''
    def __init__(self, bvh_path) -> None:
        self.bvh_path = bvh_path
        with open(bvh_path, 'r') as f:
            data = json.load(f)
            self.is_loop = data['Loop'] == 'wrap' #whether the motion is cyclic
            self.motions = np.array(data['Frames']) #original motion, data
            self.dt = self.motions[0,0]
            self.motion_time = (self.motions.shape[0] - 1) * self.dt
            self.num_frames = self.motions.shape[0] # the number of motion frames
            self.motions = self.motions[:,1:]
            self.motions_joints = {}
            self.pre_rotation = R.from_quat([0, 0.707, 0 , 0.707])
        #intialize motions for joints
        #source order of the joints
        self.source_names = ['pos_root', 'quat_root', 'quat_chest', 'quat_neck', 'quat_rhip', 'quat_rknee', 'quat_rankle', 'quat_rshoulder', 'quat_relbow'] + \
        ['quat_lhip', 'quat_lknee', 'quat_lankle', 'quat_lshoulder', 'quat_lelbow']
        data_idx = [(0,3), (3,7), (7, 11), (11,15), (15, 19), (19, 20), (20, 24), (24, 28), (28,29)] + \
            [(29, 33), (33, 34), (34,38), (38, 42), (42, 43)]
        for i in range(len(self.source_names)):
            name = self.source_names[i]
            idx = data_idx[i]
            if(idx[1] - idx[0] == 4):
                #sphere joints
                self.motions_joints[name] = {}
                self.motions_joints[name]['motion'] = np.zeros(self.motions[:, idx[0]: idx[1]].shape)
                self.motions_joints[name]['motion'][:,0] = self.motions[:, idx[0] + 1]
                self.motions_joints[name]['motion'][:,1] = self.motions[:, idx[0] + 2]
                self.motions_joints[name]['motion'][:,2] = self.motions[:, idx[0] + 3]
                self.motions_joints[name]['motion'][:,3] = self.motions[:, idx[0]]
                self.motions_joints[name]['motion'] = R.from_quat(self.motions_joints[name]['motion'].tolist())
                #compute the velocity of spherical joints
                self.motions_joints[name]['vel'] = np.zeros((self.motions.shape[0], 3))
                #embed()
                if(name != 'quat_root' and name not in ['quat_lankle', 'quat_rankle']):
                    vel = self.motions_joints[name]['motion'].as_euler('xyz').copy()
                    self.motions_joints[name]['vel'][:-1,:] = vel[1:,:] - vel[:-1,:]
                elif(name == 'quat_root'):
                    q = ( self.motions_joints[name]['motion'] * self.pre_rotation)
                    dq = q[1:] * q[:-1].inv()
                    dq = dq.as_quat()
                    dtheta = np.arccos(np.clip(dq[:,-1], -1, 1)) * 2 #shape(n,)
                    axis = dq[:,0:3] / np.sin(dtheta/2).reshape((-1,1))
                    self.motions_joints[name]['vel'][:-1,:] = axis * dtheta.reshape((-1,1))
                else:
                    vel = self.motions_joints[name]['motion'].as_euler('xyz').copy()
                    vel[:,1] = vel[:,-1]
                    vel[:,-1] = 0
                    self.motions_joints[name]['vel'][:-1,:] = vel[1:,:] - vel[:-1,:]
                self.motions_joints[name]['motion'] = Slerp([i for i in range(self.motions.shape[0])], self.motions_joints[name]['motion'])
                self.motions_joints[name]['dof'] = 3
                
            else:
                #revolute joints or root translation
                self.motions_joints[name] = {}
                self.motions_joints[name]['motion'] = np.array(self.motions[:,idx[0]: idx[1]])
                #compute the velocity of root and revolute joint
                if(name == 'pos_root'):
                    self.motions_joints[name]['vel'] = np.zeros((self.motions.shape[0], 3))
                    #embed()
                else:
                    self.motions_joints[name]['vel'] = np.zeros((self.motions.shape[0], 1))
                    #embed()
                self.motions_joints[name]['vel'][:-1,:] = np.array(self.motions[1:,idx[0]: idx[1]]) - np.array(self.motions[:-1,idx[0]: idx[1]])
                self.motions_joints[name]['dof'] = 1

        #target order of the joints
        self.target_names = ['pos_root', 'quat_root', 'quat_lhip', 'quat_lknee', 'quat_lankle', 'quat_rhip', 'quat_rknee', 'quat_rankle', 'quat_chest'] + \
        ['quat_neck', 'quat_lshoulder', 'quat_lelbow', 'quat_rshoulder', 'quat_relbow']
    
    def generate_batch(self, t):
        '''
        t: np.array([t0, t1, t2, t3,...tn])
        pos: np.array([[q0],[q1],...[qn]])
        vel: np.array([[qdot0], [qdot1],...[qdotn]])
        '''
        t = t % self.motion_time
        t1 = np.asarray(np.floor(t / self.dt), dtype = np.int)%self.num_frames
        t2 = (t1 + 1) % self.num_frames 
        pose = []
        vel = []
        for name in self.target_names:
            joint = self.motions_joints[name]
            if(joint['dof']==1):
                alpha =  ((t - t1 * self.dt)/self.dt).reshape((-1,1))
                q0 = joint['motion'][t1]
                q1 = joint['motion'][t2]
                v0 = joint['vel'][t1]
                v1 = joint['vel'][t2]
                q = alpha * q0 + (1- alpha) * q1
                v = alpha * v0 + (1- alpha) * v1
                if(q.shape[1] == 3):
                    q[:,1] += 0.09 #hack : the offset of the root joint
            elif(joint['dof']==3):
                #comupte the pose
                alpha = (t/self.motion_time) * (self.num_frames - 1)
                interp_rots = joint['motion'](alpha)
                if(name != 'quat_root' and name not in ['quat_lankle', 'quat_rankle']):
                    q = interp_rots.as_euler('xyz')
                elif(name == 'quat_root'):
                    q = (interp_rots * self.pre_rotation).as_quat()
                else:
                    q = interp_rots.as_euler('xyz')
                    q[:,1] = q[:,-1]
                    q[:,-1] = 0
                    #q = np.array([q[3], q[0], q[1], q[2]])
                #compute the velocity
                alpha =  ((t - t1 * self.dt)/self.dt).reshape((-1,1))
                v0 = joint['vel'][t1]
                v1 = joint['vel'][t2]
                v = alpha * v0 + (1- alpha) * v1
            else:
                raise NotImplementedError
            pose.append(q)
            vel.append(v)
        return np.concatenate(pose, 1), np.concatenate(vel, 1)

    def generate_single(self, t):
        t = t % self.motion_time
        t1 = (int(np.floor(t / self.dt)))%self.num_frames
        t2 = (t1 + 1) % self.num_frames 
        pose = []
        vel = []
        for name in self.target_names:
            joint = self.motions_joints[name]
            #print(name)
            #print(joint['vel'].shape)
            if(joint['dof']==1):
                alpha =  (t - t1 * self.dt)/self.dt
                q0 = joint['motion'][t1]
                q1 = joint['motion'][t2]
                v0 = joint['vel'][t1]
                v1 = joint['vel'][t2]
                q = alpha * q0 + (1- alpha) * q1
                v = alpha * v0 + (1- alpha) * v1
                if(q.shape[0] == 3):
                    q[1] += 0.07 #hack : the offset of the root joint
            elif(joint['dof']==3):
                #comupte the pose
                alpha = (t/self.motion_time) * (self.num_frames - 1)
                interp_rots = joint['motion'](alpha)
                if(name != 'quat_root' and name not in ['quat_lankle', 'quat_rankle']):
                    q = interp_rots.as_euler('xyz')
                elif(name == 'quat_root'):
                    q = (interp_rots * self.pre_rotation).as_quat()
                else:
                    q = interp_rots.as_euler('xyz')
                    q[1] = q[-1]
                    q[-1] = 0
                    #q = np.array([q[3], q[0], q[1], q[2]])
                #compute the velocity
                alpha =  (t - t1 * self.dt)/self.dt
                v0 = joint['vel'][t1]
                v1 = joint['vel'][t2]
                v = alpha * v0 + (1- alpha) * v1
            else:
                raise NotImplementedError
            pose.append(q)
            vel.append(v)
        return np.concatenate(pose), np.concatenate(vel)



