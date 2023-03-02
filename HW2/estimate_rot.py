import numpy as np
from scipy import io
from quaternion import Quaternion
from scipy import io
import scipy

#data files are numbered on the server.
#for exmaple imuRaw1.mat, imuRaw2.mat and so on.
#write a function that takes in an input number (1 through 3)
#reads in the corresponding imu Data, and estimates
#roll pitch and yaw using an unscented kalman filter
# from ukf import UKF

class UKF:
    def __init__(self, mean_0, Sigma_0, Q, R):
        self.mean_0 = mean_0        # Initial mean estimate
        self.Sigma_0 = Sigma_0      # Initial covariance in state
        self.Q = Q                  # Sigmarocess model noise
        self.R = R                  # Dynamics model noise

    def generate_sigma_points(self, Sigma_k):
        n = Sigma_k.shape[1]
        S = scipy.linalg.sqrtm(n * (Sigma_k + self.Q))
        
        S = np.hstack((S, -S))
        quat = S[:3, :]
        w = S[3:6, :]

        return quat, w

    def recompute_gaussian_from_sigma_pts(self, mean_k, Sigma_k, t_):

        q_k = Quaternion(mean_k[0], mean_k[1:4])
        w_k = mean_k[4:7]
       
        quat, w = self.generate_sigma_points(Sigma_k)
        n = w.shape[1]

        state_i = np.zeros((7, 12))
        q_diff = Quaternion()
        for i in range(0, n):
            qw = Quaternion()
            qw.from_axis_angle(a=quat[:, i].reshape(-1))
            qk1 = q_k.__mul__(qw)
            q_diff.from_axis_angle(a=w_k*t_)
            q_k_ = qk1.__mul__(q_diff)
            y_i = np.hstack([q_k_.q, w_k+w[:, i]])
            state_i[:, i] = y_i

        return state_i.T
    
    def gradient_descent(self, q_i, init):
        q_mean = init

        thresh = 0.01
        eps = 1e-4
        error_i = np.zeros((q_i.shape[0], 3)) 
        iter = 0
        e_mean_quat_obj = Quaternion()

        while True:
            for i in range(0, q_i.shape[0]):
                q = Quaternion(scalar=q_i[i, 0], vec=q_i[i, 1:4])
                q_calc = q.__mul__(q_mean.inv())
                q_calc_ = q_calc.scalar()

                if q_calc_ > 1.0 and abs(q_calc_ - 1.0) < eps:
                    q_calc = Quaternion(1.0, q_calc.vec())
                
                elif q_calc_ < -1.0 and abs(q_calc_ + 1.0) < eps:
                    q_calc = Quaternion(-1.0, q_calc.vec())

                error_i[i, :] = q_calc.axis_angle()

            error_mean = np.mean(error_i, axis=0)
            e_mean_quat_obj.from_axis_angle(error_mean)
            q_mean = e_mean_quat_obj.__mul__(q_mean)

            if np.all(abs(error_mean) < thresh) or iter>100:
                break
            iter = iter+1

        return q_mean, error_i

    def propagate(self, mean_k, Sigma_k, t_):
        state_i = self.recompute_gaussian_from_sigma_pts(mean_k, Sigma_k, t_)

        q_init = Quaternion(scalar=state_i[0,0], vec=state_i[0, 1:4].reshape(3,))
        q_mean, error_i = self.gradient_descent(state_i[:, 0:4], init=q_init)

        w_i = state_i[:, 4:7] #covariance of state at timestep i
        w_w_bar = w_i - np.mean(w_i, axis=0)

        mean_hat = np.hstack([q_mean.q, np.mean(w_i, axis=0)])
        w_i_hat = np.hstack([error_i, w_w_bar])
        Sigma_k_ = (w_i_hat.T @ w_i_hat)/(w_i_hat.shape[0])

        return mean_hat, Sigma_k_, state_i, w_i_hat

    def measurement_model(self, x_k):
        q_k = Quaternion(x_k[0], x_k[1:4])
        g = Quaternion(0, [0, 0, 9.81])
        g_ = (q_k.inv()).__mul__(g.__mul__(q_k))

        Z = np.hstack([g_.vec(), x_k[4:7]])

        return Z
    
    def compute_measurement_model(self, state_i):
        n = state_i.shape[0]
        Z_i = np.zeros((n, 6))

        for i in range(0, n):
            Z_i[i] = self.measurement_model(state_i[i])

        return Z_i
        
    def compute_covariance_matrices(self, Z_i, w_i_hat):
        z_mean = np.mean(Z_i, axis=0)
        Sigma_zz = ((Z_i-z_mean).T @ (Z_i-z_mean))/12
        Sigma_yy = Sigma_zz + self.R
        Sigma_xz = (w_i_hat.T @ (Z_i-z_mean))/w_i_hat.shape[0]

        return Sigma_yy, Sigma_xz

    def compute_Kalman_Gain(self, state_i, w_i_hat):
        
        Z_i = self.compute_measurement_model(state_i)   
        Sigma_yy, Sigma_xz = self.compute_covariance_matrices(Z_i, w_i_hat)
        Kalman_gain = Sigma_xz @ np.linalg.inv(Sigma_yy)

        return Kalman_gain
    
    def compute_innovation(self, state_i, measurements_array):
        Z_i = self.compute_measurement_model(state_i)   
        z_mean = np.mean(Z_i, axis=0)        
        innovation = measurements_array - z_mean
        return innovation

    def update(self, mean_k, Sigma_k, measurements_array, state_i, Kalman_gain, w_i_hat):

        Z_i = self.compute_measurement_model(state_i)
        Sigma_yy, _ = self.compute_covariance_matrices(Z_i, w_i_hat)
        innovation = self.compute_innovation(state_i, measurements_array)

        mu_d = Kalman_gain @ innovation
        mu_d_ = Quaternion()
        mu_d_.from_axis_angle(mu_d[0:3])
        mu_d_x = mu_d_.__mul__(Quaternion(mean_k[0], mean_k[1:4]))
        mu_d_w = mean_k[4:7] + mu_d[3:6]
        mean_k = np.hstack([mu_d_x.q, mu_d_w])

        Sigma_k = Sigma_k - Kalman_gain @ Sigma_yy @ Kalman_gain.T

        return mean_k, Sigma_k

    def run_ukf(self, accel, gyro, imu_ts, T):
        mean_k = self.mean_0
        Sigma_k = self.Sigma_0
        roll, pitch, yaw = [], [], []
        mean, Sigma = [], []

        q_init = Quaternion(mean_k[0], mean_k[1:4])
        euler_angles_init = q_init.euler_angles()

        # Store init states
        roll.append(euler_angles_init[0])
        pitch.append(euler_angles_init[1])
        yaw.append(euler_angles_init[2])

        mean.append(mean_k)
        Sigma.append(Sigma_k)

        iter = 0
        for t in range(1, T):
            t_ = imu_ts[t]-imu_ts[t-1]
            mean_k, Sigma_k, state_i, w_i_hat = self.propagate(mean_k, Sigma_k, t_)
            measurements_array = np.hstack([accel[iter], gyro[iter]])

            Kalman_gain = self.compute_Kalman_Gain(state_i, w_i_hat)

            mean_k, Sigma_k = self.update(mean_k, Sigma_k, measurements_array, state_i, Kalman_gain, w_i_hat)

            euler_angles_ = Quaternion(mean_k[0], mean_k[1:4])

            euler = euler_angles_.euler_angles()
            roll.append(euler[0])
            pitch.append(euler[1])
            yaw.append(euler[2])
            mean.append(mean_k)
            Sigma.append(Sigma_k)
            iter = iter+1

        return roll, pitch, yaw, mean, Sigma


def calibrate(arr, parameter):
    if parameter == 'accel':
        # beta = np.mean(arr, axis = 1)
        # beta[2] = np.sum(beta[:2])/2
        # M = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
        # tmp = M @ (arr - beta.reshape(-1,1))
        # alpha = np.mean(tmp[2, :100]) / 9.81
        # return alpha, beta
        alpha = 35
        beta = np.array([505, 505, 505])[:, None]
        arr = (arr - beta) * 3300 / (1023 * alpha)
        arr[0] = -1 * arr[0]
        arr[1] = -1 * arr[1]        
        arr = arr.T

    elif parameter == 'gyro':
        # beta = np.mean(arr, axis = 1)
        # M = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
        # tmp = (M @ arr) - beta.reshape((-1,1))
        # tmp_normalised = np.abs(tmp /1e-6)
        # alpha = np.mean(tmp_normalised[tmp_normalised < 1e-3])
        # return alpha, beta
        alpha = 200
        beta = np.array([370, 370, 370])[:, None]
        arr = (arr - beta) * 3300 / (1023 * alpha)
        arr = np.vstack([arr[1], arr[2], arr[0]]).T

    return arr

def estimate_rot(data_num=1):
    imu = io.loadmat('imu/imuRaw' + str(data_num) + '.mat')
    accel = imu['vals'][0:3,:]
    gyro = imu['vals'][3:6,:]
    imu_ts = imu['ts']
    T = np.shape(imu['ts'])[1]
    # vicon = io.loadmat('vicon/viconRot' + str(data_num) + '.mat')

    # your code goes here

    q_init = Quaternion()
    q_init = q_init.q
    w_init = 0.5 * np.ones(3)

    mean_0 = np.hstack([q_init, w_init])
    Sigma_0 = 0.3 * np.diag(np.ones(6))
    Q = 0.1 * np.diag(np.ones(6))
    R = 3 * np.diag(np.ones(6))

    accel = calibrate(accel, 'accel')
    gyro = calibrate(gyro, 'gyro')

    ukf = UKF(mean_0, Sigma_0, Q, R)
    state = ukf.run_ukf(accel, gyro, imu_ts[0, :], T)
    roll, pitch, yaw, mean, Sigma = state
    mean, Sigma = np.array(mean), np.array(Sigma)

    return roll,pitch,yaw

