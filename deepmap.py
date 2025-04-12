import deepcell
from config import get_parse_args
import torch
import os
from typing import List
import sys  # 导入sys模块
sys.setrecursionlimit(100000)  # 将默认的递归深度修改为3000

def extract_fanout(node_index, g):
    fanout = []
    current_level = g.forward_level[node_index].item()  # 获取当前节点的level
    edges = g.edge_index.t()
    for edge in edges:
        if node_index in edge:
            neighbor = edge[1].item() if edge[0].item() == node_index else edge[0].item()
            if g.forward_level[neighbor].item() > current_level:
                fanout.append(neighbor)
            #print("edge =", edge)
    
    return list(set(fanout))  # 返回唯一的扇出节点列表

def get_all_aig_files(directory: str) -> List[str]:
    """
    获取目录下（包括子目录）所有 .aig 文件的绝对路径，并存入列表返回
    
    Args:
        directory (str): 要搜索的根目录
        
    Returns:
        List[str]: 包含所有 .aig 文件绝对路径的列表
    """
    aig_file_paths = []  # 存储所有 .aig 文件的绝对路径
    
    # 使用 os.walk 递归遍历目录
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.aig'):
                # 构造绝对路径并添加到列表
                abs_path = os.path.abspath(os.path.join(root, file))
                aig_file_paths.append(abs_path)
    
    return aig_file_paths


if __name__ == "__main__":
    target_dir = '/home/jwt/DeepCell/opt'  # 替换为你的目标目录
    output_dir = '/home/jwt/DeepCell/opttxt/'
    input_paths = get_all_aig_files(target_dir)
    args = get_parse_args()
    #Create MixGate
    model = deepcell.TopModel(
        args,
        dc_ckpt = '/home/jwt/DeepCell/ckpt/dc.pth',
        dg_ckpt = '/home/jwt/DeepCell/ckpt/dg.pth' )    
    model.load('/home/jwt/DeepCell/exp/train/train/model_last.pth') #deepcell pth
    #model.load_state_dict(torch.load('/home/jwt/DeepCell/ckpt/DeepMap_model.pth', map_location='cuda:0') )   # Load pretrained model
    parser = deepcell.AigParser()   # Create AigParser
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    for input_path in input_paths:
        print("parsing aig file =", input_path)
        graph = parser.read_aiger(input_path) # Parse AIG into Graph
        # if len(graph.x > 10000):
        #     continue
        # print("graph.gate = ", graph.gate)
        print("graph.edge_index =", graph.edge_index.shape)
        _, hf, hs, without_pi = model(graph)       # Model inference 
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

        gate_dic = {}
        for index, item in enumerate(graph.forward_level):
            gate_dic[index] = extract_fanout(index, graph)

        gate_dictionary = {}
        for key, value in gate_dic.items():
            val = []
            # 只有当当前门是类型1且value列表不为空时才处理
            if value and graph.gate[key] == 1:
                for gate_index in value:
                    # 如果子门是类型1，直接添加
                    if graph.gate[gate_index] == 1:
                        val.append(gates[gate_index])
                    # 如果子门是类型2，添加它的所有子门
                    elif graph.gate[gate_index] == 2:
                        # 需要确保gate_index在gate_dic中存在
                        if gate_index in gate_dic:
                            for v in gate_dic[gate_index]:
                                val.append(gates[v])
                # 更新字典
                gate_dictionary[key] = val 

        output_name = input_path.split("/")[-1].split(".")[0] + ".txt"
        output_path = os.path.join(output_dir, output_name)
        print("output_path =", output_path)
        with open(output_path, 'w') as f:
            for index, value in gate_dictionary.items():
                if index in without_pi:
                    # 先转成字符串，再去除括号
                    value_str = ' '.join(value)  # 自动用空格连接元素，无逗号/引号
                    f.write(f"{int(index)+1} {value_str}\n")

    # print(f"数据已写入: {full_path}")
    # print("Graph.forward_level", graph.forward_level)
    # print("G.egde_index", graph.edge_index.t())
    # print("gates =", gates)
    # hs: structural embeddings, hf: functional embeddings
    # hs/hf: [N, D]. N: number of gates, D: embedding dimension (default: 128)


