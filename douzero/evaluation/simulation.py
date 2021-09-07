import multiprocessing as mp
import pickle
import douzero.env.env
from douzero.dmc.models import Model
from douzero.env.game import GameEnv
import torch
import numpy as np
import BidModel

def load_card_play_models(card_play_model_path_dict):
    players = {}

    for position in ['landlord', 'landlord_up', 'landlord_down']:
        if card_play_model_path_dict[position] == 'rlcard':
            from .rlcard_agent import RLCardAgent
            players[position] = RLCardAgent(position)
        elif card_play_model_path_dict[position] == 'random':
            from .random_agent import RandomAgent
            players[position] = RandomAgent()
        else:
            from .deep_agent import DeepAgent
            players[position] = DeepAgent(position, card_play_model_path_dict[position])
    return players

def mp_simulate(card_play_data_list, card_play_model_path_dict, q, output, bid_output, title):
    players = load_card_play_models(card_play_model_path_dict)
    EnvCard2RealCard = {3: '3', 4: '4', 5: '5', 6: '6', 7: '7',
                        8: '8', 9: '9', 10: 'T', 11: 'J', 12: 'Q',
                        13: 'K', 14: 'A', 17: '2', 20: 'X', 30: 'D'}
    env = GameEnv(players)
    bid_model = None
    if bid_output:
        model = Model(device=0)
        bid_model = model.get_model("bidding")
        bid_model_path = card_play_model_path_dict["landlord"].replace("landlord", "bidding")
        weights = torch.load(bid_model_path)
        bid_model.load_state_dict(weights)
        bid_model.eval()
    for idx, card_play_data in enumerate(card_play_data_list):
        env.card_play_init(card_play_data)
        if bid_output:
            output = True
            bid_results = []
            bid_values = []
            bid_info_list = [
                np.array([[-1,-1,-1],
                          [-1,-1,-1],
                          [-1,-1,-1],
                          [-1,-1,-1]]),
                np.array([[0,0,0],
                          [-1,-1,-1],
                          [-1,-1,-1],
                          [-1,-1,-1]]),
                np.array([[1,0,0],
                          [-1,-1,-1],
                          [-1,-1,-1],
                          [-1,-1,-1]]),
                np.array([[0,0,0],
                          [0,0,0],
                          [-1,-1,-1],
                          [-1,-1,-1]]),
                np.array([[0,0,1],
                          [1,0,0],
                          [-1,-1,-1],
                          [-1,-1,-1]]),
                np.array([[0,1,0],
                          [0,0,1],
                          [1,0,0],
                          [-1,-1,-1]]),
            ]
            for bid_info in bid_info_list:
                bid_obs = douzero.env.env._get_obs_for_bid(1, bid_info, card_play_data["landlord"])
                result = bid_model.forward(torch.tensor(bid_obs["z_batch"], device=torch.device("cuda:0")), torch.tensor(bid_obs["x_batch"], device=torch.device("cuda:0")), True)
                values = result["values"]
                bid = 1 if values[1] > values[0] else 0
                bid_results.append(bid)
                bid_values.append(values[bid])
            result2 = BidModel.predict_env(card_play_data["landlord"])
            print("".join([EnvCard2RealCard[c] for c in card_play_data["landlord"]]), end="")
            print(" bid: %i|%i%i|%i%i|%i (%.3f %.3f %.3f %.3f %.3f %.3f) %.1f" % (bid_results[0],bid_results[1],bid_results[2],bid_results[3],bid_results[4],bid_results[5],bid_values[0],bid_values[1],bid_values[2],bid_values[3],bid_values[4],bid_values[5], result2))
        if output and not bid_output:
            print("\nStart ------- " + title)
            print ("".join([EnvCard2RealCard[c] for c in card_play_data["landlord"]]))
            print ("".join([EnvCard2RealCard[c] for c in card_play_data["landlord_down"]]))
            print ("".join([EnvCard2RealCard[c] for c in card_play_data["landlord_up"]]))
        # print(card_play_data)
        count = 0
        while not env.game_over and not bid_output:
            action = env.step()
            if output:
                if count % 3 == 2:
                    end = "\n"
                else:
                    end = "   "
                if len(action) == 0:
                    print("Pass", end=end)
                else:
                    print("".join([EnvCard2RealCard[c] for c in action]), end=end)
                count+=1
        if idx % 10 == 0 and not bid_output:
            print("\nindex", idx)
        # print("End -------")
        env.reset()

    q.put((env.num_wins['landlord'],
           env.num_wins['farmer'],
           env.num_scores['landlord'],
           env.num_scores['farmer']
           ))

def data_allocation_per_worker(card_play_data_list, num_workers):
    card_play_data_list_each_worker = [[] for k in range(num_workers)]
    for idx, data in enumerate(card_play_data_list):
        card_play_data_list_each_worker[idx % num_workers].append(data)

    return card_play_data_list_each_worker

def evaluate(landlord, landlord_up, landlord_down, eval_data, num_workers, output, output_bid, title):

    with open(eval_data, 'rb') as f:
        card_play_data_list = pickle.load(f)

    card_play_data_list_each_worker = data_allocation_per_worker(
        card_play_data_list, num_workers)
    del card_play_data_list

    card_play_model_path_dict = {
        'landlord': landlord,
        'landlord_up': landlord_up,
        'landlord_down': landlord_down}

    num_landlord_wins = 0
    num_farmer_wins = 0
    num_landlord_scores = 0
    num_farmer_scores = 0

    ctx = mp.get_context('spawn')
    q = ctx.SimpleQueue()
    processes = []
    for card_paly_data in card_play_data_list_each_worker:

        p = ctx.Process(
                target=mp_simulate,
                args=(card_paly_data, card_play_model_path_dict, q, output, output_bid, title))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    for i in range(num_workers):
        result = q.get()
        num_landlord_wins += result[0]
        num_farmer_wins += result[1]
        num_landlord_scores += result[2]
        num_farmer_scores += result[3]

    num_total_wins = num_landlord_wins + num_farmer_wins
    print('WP results:')
    print('landlord : Farmers - {} : {}'.format(num_landlord_wins / num_total_wins, num_farmer_wins / num_total_wins))
    print('ADP results:')
    print('landlord : Farmers - {} : {}'.format(num_landlord_scores / num_total_wins, 2 * num_farmer_scores / num_total_wins)) 
