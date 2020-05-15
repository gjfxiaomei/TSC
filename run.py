import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from simulation import Simulation
from utils import set_sumo,set_save_path
from parseargs import parse_cl_args
from networkloader import NetworkLoader
from visualization import Visualization
import datetime
from logger import Logger

if __name__ == "__main__":
    args = parse_cl_args()
    # sumo_cmd = set_sumo(args.gui, args.sumocfg, args.max_steps, args.port)

    log = Logger('episode_info.log',level='info')

    netloader = NetworkLoader(args.roadnet)
    netdata = netloader.get_net_data()
    Simulation = Simulation(args, netdata, log)
    episode = 0

    timestamp_start = datetime.datetime.now()

    while episode < args.total_episodes:
        # print("episode :",episode + 1)
        epsilon = 1.0 - (episode / args.total_episodes)
        simulation_time,training_time = Simulation.run(episode,epsilon)
        log.logger.info('episode:{} - Simulation time:{}s - Training time:{}s - Total:{}s'.format(episode+1, simulation_time, training_time,round(simulation_time+training_time, 1)))
        # print('Simulation time:', simulation_time, 's - Training time:', training_time, 's - Total:', round(simulation_time+training_time, 1), 's')
        episode += 1
        if args.mode == 'train' and args.tsc == 'dqn':
            Simulation.save_model()
    
    Simulation.close()

    log.logger.info("----- Start time:{}".format(timestamp_start))
    log.logger.info("----- End time:{}".format(datetime.datetime.now()))

    Simulation.visualize()
    
    

    # Visualization.save_data_and_plot(data=Simulation.avg_queue_length_store, filename='queue', xlabel='Episode', ylabel='Average queue length (vehicles)')

