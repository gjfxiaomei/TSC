import numpy as np
import timeit
import os,sys,subprocess
from dqn import DqnAgent
from baseline.uniform_controller import UniformController
from baseline.dqn_controller import DqnController
from tlcontroller import TLController
from vehiclegen import VehicleGen
from saver import Saver
from utils import set_sumo,set_save_path

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
    from sumolib import checkBinary
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

import traci

# phase codes based on environment.net.xml
PHASE_NS_GREEN = 0  # action 0 code 00
PHASE_NS_YELLOW = 1
PHASE_NSL_GREEN = 2  # action 1 code 01
PHASE_NSL_YELLOW = 3
PHASE_EW_GREEN = 4  # action 2 code 10
PHASE_EW_YELLOW = 5
PHASE_EWL_GREEN = 6  # action 3 code 11
PHASE_EWL_YELLOW = 7

class Simulation:
    def __init__(self,args,netdata,log):

        self.t = 0
        self.args = args
        self.sumo_cmd = set_sumo(args.gui, args.roadnet, args.max_steps, args.port)
        self.log = log
        print(self.sumo_cmd)

        self.save_path = set_save_path(args.roadnet,args.tsc)
        self.saver = Saver(self.save_path)

        self.max_steps = args.max_steps
        self.green_t = args.green_duration
        self.yellow_t = args.yellow_duration
        self.red_t = args.red_duration
        self.mode = args.mode
        self.scale = args.scale
        self.port = args.port
        self.netdata = netdata
        self.tl_ids = self.netdata['inter'].keys()
        self.sumo_process = subprocess.Popen(self.sumo_cmd)
        self.conn = traci.connect(self.port)

        self.netdata = self.update_netdata()
        self.vehiclegen = VehicleGen(self.netdata,self.conn,self.mode,self.scale,self.max_steps)


        self.Controllers = {str(id):None for id in self.tl_ids}
        self.create_controllers(self.args.tsc)
        if self.args.tsc == 'dqn' and self.args.tsc == 'test':
            self.load_model()
        self.v_start_times = {}
        self.v_travel_times = {}
        self.episode_performance = []
        self.set_item_path()

        
    def create_controllers(self,tsc):
        if tsc == 'uniform':
            for id in self.tl_ids:
                c = UniformController(self.conn, id, self.netdata, self.mode,self.red_t,self.yellow_t,self.green_t)
                self.Controllers[id] = c
        elif tsc == 'dqn':
            for id in self.tl_ids:
                # density and queue and current phase, +1 for all-red clearance pahse.
                state_size = len(self.netdata['inter'][id]['incoming_lanes'])*2 + len(self.netdata['inter'][id]['green_phases']) + 1
                action_size = len(self.netdata['inter'][id]['green_phases'])
                print(state_size, action_size)
                rlagent = DqnAgent(self.args.batch_size,state_size,action_size)
                c = DqnController(self.conn, id, self.netdata, self.mode, rlagent, self.green_t, self.yellow_t, self.red_t)
                self.Controllers[id] = c

    def run(self,episode,epsilon):
        start_time = timeit.default_timer()

        # set epsilon
        if self.mode == 'train' and self.args.tsc == 'dqn':
            for t in self.Controllers:
                self.Controllers[t].rlagent.set_epsilon(epsilon)
        # reset the schedule
        self.vehiclegen.reset()
        print("Simulating...")

        self.t = 0

        while self.t < self.max_steps:
            if self.vehiclegen:
                self.vehiclegen.run()

            self.update_travel_times()
            # print(self.v_travel_times)
            for t in self.Controllers:
                self.Controllers[t].run()
            self.sim_step()
        
        simulation_time = round(timeit.default_timer() - start_time, 1)

        self.episode_performance.append(self.get_performance())
            
        training_time = 0
        if self.mode == 'train' and self.args.tsc == 'dqn':
            print("Training...")
            start_time = timeit.default_timer()
            for _ in range(800):
                for t in self.Controllers:
                    self.Controllers[t].train()
            training_time = round(timeit.default_timer() - start_time, 1)
        
        return simulation_time, training_time
     
    def sim_step(self):
        self.conn.simulationStep()
        self.t += 1

    def get_performance(self):
        tt =self.get_travel_times()
        if len(tt) > 0:
            # return [str(int(np.mean(tt))), str(int(np.std(tt)))]
            return np.mean(tt)
        else:
            return [str(int(0.0)), str(int(0.0))]

    def get_travel_times(self):
        return [self.v_travel_times[v] for v in self.v_travel_times]

    def update_travel_times(self):
        # record vehicles depart and arrive time to calc the travel time.
        for v in self.conn.simulation.getDepartedIDList():
            self.v_start_times[v] = self.t

        for v in self.conn.simulation.getArrivedIDList():
            self.v_travel_times[v] = self.t - self.v_start_times[v]
            del self.v_start_times[v]

    def update_netdata(self):
        tcs = {tl_id:TLController(self.conn,tl_id,self.netdata,self.mode,self.red_t,self.yellow_t) for tl_id in self.tl_ids }
        for tl in tcs:
            self.netdata['inter'][tl]['incoming_lanes'] = tcs[tl].incoming_lanes
            self.netdata['inter'][tl]['green_phases'] = tcs[tl].green_phases
        return self.netdata
        

    def _save_episode_stats(self):
        """
        Save the stats of the episode to plot the graphs at the end of the session
        """
        # self._reward_store.append(self._sum_neg_reward)  # how much negative reward in this episode

        for id in self.tl_ids:
            self._cumulative_wait_store[id].append(self._sum_waiting_time[id])
        print(self._cumulative_wait_store)

        for id in self.tl_ids:
            self._avg_queue_length_store[id].append(self._sum_queue_length[id] /self._max_steps)
        print(self._avg_queue_length_store)
        # self._avg_queue_length_store.append(self._sum_queue_length / self._max_steps)  # average number of queued cars per step, in this episode
    
    def set_item_path(self):
        for t in self.Controllers:
            item_path = os.path.join(self.save_path,t,'')
            os.makedirs(os.path.dirname(item_path),exist_ok=True)

    def save_model(self):
        for t in self.Controllers:
            item_path = os.path.join(self.save_path,t)
            self.Controllers[t].rlagent.save_model(item_path)
    
    def load_model(self):
        for t in self.Controllers:
            item_path = os.path.join(self.save_path,t)
            self.Controllers[t].rlagent.load_model(item_path)

    def close(self):
        #self.conn.close()
        self.conn.close()
        self.sumo_process.terminate()
    
    def save_result(self):
        for id in self.tl_ids:
            item_path = os.path.join(self.save_path,id,'')
            self.saver.set_path(item_path)
            self.saver.save_data_and_plot(data=self.episode_performance, filename=self.args.mode+'-'+'Travel time of '+str(id), xlabel='Episode', ylabel='episode mean travel time (s)')

