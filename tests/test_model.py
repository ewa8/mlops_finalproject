import src.models.model as md
import torch


def test_modelshape(): 
    model = md.TumorClassifier()
    model.eval()
    expected_output = (50,1)
    with torch.no_grad():
        input_shape = torch.randn((50, 3, 224, 224))    
        out_shape = model.forward(input_shape) 
    
    assert len(out_shape) == expected_output[0]
    assert expected_output == (len(out_shape), 1)
