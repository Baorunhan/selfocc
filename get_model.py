import pickle
import torch
import sys
import yaml
import os
from selfmodel.Net import Net
from selfmodel.CustomeFPN import *
from selfmodel.ResNet import *
from selfmodel.view_transformer import LSSViewTransformer
from selfmodel.bev_encoder_FPN import bev_encoder_with_FPN
from selfmodel.bev_occ_head import BEVOCCHead2D
 
sys.path.append(os.path.dirname(os.path.abspath(__file__))) 
# from structure.model.backbones import ResNet
# from structure.model.fpn import CustomFPN

class ModelWrapper(nn.Module):
    def __init__(self, original_model):
        super().__init__()
        self.model = original_model
    
    def forward(self, img_inputs, img_metas, points):
        """
        包装原始模型的复杂前向传播
        
        假设原始调用：
        model(return_loss=False, rescale=True, **data)
        
        其中 data 包含 'img' 键
        """
        data = {
            'img_inputs': img_inputs,  
            'img_metas': img_metas,
            'points': points
        }
        result = self.model(**data)

        # 提取需要的输出，假设结果在 result[0]
        # if isinstance(result, (list, tuple)):
        #     return result[0]
        return result


def main():
    pass


if __name__ == '__main__':
    
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)  
    
    with open('dummy_data.pkl', 'rb') as f:  # with语句自动关闭文件，更安全
        data_input = pickle.load(f)
        
    data = {}    
    data.update({"img_inputs":[data_input['img_inputs'][0][0].cuda(),
                        data_input['img_inputs'][0][1].cuda(),
                        data_input['img_inputs'][0][2].cuda(),
                        data_input['img_inputs'][0][3].cuda(),
                        data_input['img_inputs'][0][4].cuda(),
                        data_input['img_inputs'][0][5].cuda(),
                        data_input['img_inputs'][0][6].cuda()]})
    
    # data.update({"img_metas": data_input['img_metas'][0]})
    # data_input['points'][0].data[0][0] = data_input['points'][0].data[0][0].cuda()
    # data.update({"points": data_input['points'][0]})
    # 根据实际情况调整
    data.update({'img_metas': [[{
        'flip': False,
        # 其他必要的元数据
    },{
        'flip': False,
        # 其他必要的元数据
    },{
        'flip': False,
        # 其他必要的元数据
    },{
        'flip': False,
        # 其他必要的元数据
    },{
        'flip': False,
        # 其他必要的元数据
    },{
        'flip': False,
        # 其他必要的元数据
    },{
        'flip': False,
        # 其他必要的元数据
    }]]})
    data.update({'points': [[torch.tensor([1,2,3,4,5]).cuda()]]})  # 根据实际情况调整

    ### 前馈 网络 ###
    
    grid_config = {
    'x': [-40, 40, 0.4],
    'y': [-40, 40, 0.4],
    'z': [-1, 5.4, 6.4],
    'depth': [1.0, 45.0, 0.5],
    }
      
    model_predict = Net(
        training=False,
        #主干提取网络
        backbone=resnet18(config),
        #FPN金字塔
        neck=CustomeFPN_V1(config),
        #视角转换
        img_view_transformer=LSSViewTransformer(
        out_channels=64,
        in_channels=256,                    
        input_size=(256, 704), 
        grid_config=grid_config,
        collapse_z=True,
        downsample=16
        ),
        #bev主干网络
        img_bev_encoder=bev_encoder_with_FPN(
        numC_input=64,
        in_channels=64 * 2 + 64 * 8,
        num_channels=[64 * 2, 64 * 4, 64 * 8],
        num_layer=[2, 2, 2]
        ),
        #占据头
        occ_head=BEVOCCHead2D(
        in_dim=256,
        out_dim=256,
        Dz=16,
        use_mask=True,
        num_classes=18,
        use_predicter=True,
        class_balance=False
        )
        ).to('cuda')
        

    model = ModelWrapper(model_predict)
    model.eval()
    # result = model(torch.randn(1,6,3,256,704).to('cuda'))  # 输入张量根据实际情况调整
    # 导出并验证
    dummy_input = torch.randn(1,6,3,256,704) # 无用
    torch.onnx.export(
        model,
        (data["img_inputs"],data["img_metas"],data["points"]),
        "complex_model.onnx",
        opset_version=13,
        do_constant_folding=False,
        input_names=['img_inputs','img_metas','points'],
        output_names=['output']
    )

    # # 验证导出的ONNX模型
    import onnx
    onnx_model = onnx.load("complex_model.onnx")
    
    for init in onnx_model.graph.initializer:
        if init.dims == [1, 1, 1, 1, 1]:
            print(f"发现可疑常量: {init.name}")
    
    onnx.checker.check_model(onnx_model)
    print("模型验证成功！")
    print(f"输入: {onnx_model.graph.input[0].name}")
    print(f"输出: {onnx_model.graph.output[0].name}")
    
    for node in onnx_model.graph.node:
        if node.name == "Reshape_386" or node.op_type == "Reshape":
            print("找到 Reshape 节点:")
            print(f"  名称: {node.name}")
            print(f"  输入: {node.input}")
            print(f"  输出: {node.output}")
            
            # 查看该节点的属性
            for attr in node.attribute:
                print(f"  属性 {attr.name}: {attr}")
