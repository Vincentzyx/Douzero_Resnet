import torch
import numpy as np

from douzero.env.env import get_obs

def _load_model(position, model_path, model_type):
    from douzero.dmc.models import model_dict_new, model_dict
    model = None
    if model_type == "general":
        model = model_dict_new[position]()
    else:
        model = model_dict[position]()
    model_state_dict = model.state_dict()
    if torch.cuda.is_available():
        pretrained = torch.load(model_path, map_location='cuda:0')
    else:
        pretrained = torch.load(model_path, map_location='cpu')
    pretrained = {k: v for k, v in pretrained.items() if k in model_state_dict}
    model_state_dict.update(pretrained)
    model.load_state_dict(model_state_dict)
    # torch.save(model.state_dict(), model_path.replace(".ckpt", "_nobn.ckpt"))
    if torch.cuda.is_available():
        model.cuda()
    model.eval()
    return model

class DeepAgent:

    def __init__(self, position, model_path):
        self.model_type = "general" if "resnet" in model_path else "old"
        self.model = _load_model(position, model_path, self.model_type)
        self.EnvCard2RealCard = {3: '3', 4: '4', 5: '5', 6: '6', 7: '7',
                            8: '8', 9: '9', 10: 'T', 11: 'J', 12: 'Q',
                            13: 'K', 14: 'A', 17: '2', 20: 'X', 30: 'D'}
    def act(self, infoset):
        if len(infoset.legal_actions) == 1:
            return infoset.legal_actions[0]

        obs = get_obs(infoset, self.model_type == "general")

        z_batch = torch.from_numpy(obs['z_batch']).float()
        x_batch = torch.from_numpy(obs['x_batch']).float()
        if torch.cuda.is_available():
            z_batch, x_batch = z_batch.cuda(), x_batch.cuda()
        y_pred = self.model.forward(z_batch, x_batch, return_value=True)['values']
        y_pred = y_pred.detach().cpu().numpy()

        best_action_index = np.argmax(y_pred, axis=0)[0]
        best_action = infoset.legal_actions[best_action_index]
        # action_list = []
        # output = ""
        # for i, action in enumerate(y_pred):
        #     action_list.append((y_pred[i].item(), "".join([self.EnvCard2RealCard[ii] for ii in infoset.legal_actions[i]]) if len(infoset.legal_actions[i]) != 0 else "Pass"))
        # action_list.sort(key=lambda x: x[0], reverse=True)
        # value_list = []
        # for action in action_list:
        #     output += str(round(action[0],3)) + " " + action[1] + "\n"
        #     value_list.append(action[0])
        # # print(value_list)
        # print(output)
        # print("--------------------\n")
        return best_action
