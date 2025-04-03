import deepcell
from config import get_parse_args
import torch

args = get_parse_args()
#Create MixGate
model = deepcell.TopModel(
    args,
    dc_ckpt = '/home/jwt/DeepCell/ckpt/dc.pth',
    dg_ckpt = '/home/jwt/DeepCell/ckpt/dg.pth' )    

model.load('/home/jwt/DeepCell/exp/train/train/model_last.pth') #deepcell pth
#model.load_state_dict(torch.load('/home/jwt/DeepCell/ckpt/DeepMap_model.pth', map_location='cuda:0') )   # Load pretrained model
parser = deepcell.AigParser()   # Create AigParser
graph = parser.read_aiger('/home/jwt/downstream_task/examples/aig_folder/adder_8_44.aig') # Parse AIG into Graph
_, hf, hs = model(graph)       # Model inference 
gate_type = model.pred_gate(hf)
gate_dic = deepcell.one_hot_mapping
i = 0
gates = []
for key, value in gate_dic.items():
    gate_dic[key] = i
    i += 1
_, predicted = torch.max(gate_type.data, 1)
predicted = predicted.tolist()
reverse_dict = {v: k for k, v in gate_dic.items()}
for i in predicted:
    gates.append(reverse_dict[i])
print("gates =", gates)
# hs: structural embeddings, hf: functional embeddings
# hs/hf: [N, D]. N: number of gates, D: embedding dimension (default: 128)


