# DNN-parser
DNN parser is a tool that allows deep neural networks to be instantiated in both tensorflow and pytorch from a configuration json file. 
<br>The simplicity of this approach makes the creation of DNNs for training accessible to all. 
<br>The proposed version is still being developed and updated, but it allows the instance of the neural network to be created without any problem. 
<br><br>The following sections will describe the organisation of the json to be submitted to the parser, as well as a summary of the codes describing each layer. 

## Json structure
The structure of the configuration json is very simple and requires the following keys <br>


|JSON key |Value| Description|
| :---:   | :---: | : ---: |
| `input_shape` | aaa| a nested |


<details>  

  <summary>An example of json strucuture of a Unet architecture is here reported.</summary>
  
```
{
    "input_shape": {
        "input_shape0": 224,
        "input_shape1": 224,
        "input_shape2": 3,
        "label_shape": 1
    },
    "architecture": {
        "1": "doubleConv",
        "2": "Maxpool2d",
        "3": "doubleConv",
        "4": "Maxpool2d",
        "5": "doubleConv",
        "6": "Maxpool2d",
        "7": "doubleConv",
        "8": "Maxpool2d",
        "9": "ElbowUNet",
        "10": "Upsample",
        "11": "UpDoubleConv",
        "12": "Upsample",
        "13": "UpDoubleConv",
        "14": "Upsample",
        "15": "UpDoubleConv",
        "16": "Upsample",
        "17": "UpDoubleConv_out",
        "18": "cnn2d"
    },
    "convolutional": {
        "cn1": 64,
        "cn2": 128,
        "cn3": 256,
        "cn4": 512,
        "cn5": 1024,
        "cn6": 512,
        "cn7": 256,
        "cn8": 128,
        "cn9": 64, 
        "cn10": 1
    },
    "kernel": {
        "kernel1": 3,
        "kernel2": 3,
        "kernel3": 3,
        "kernel4": 3,
        "kernel5": 3,
        "kernel6": 3,
        "kernel7": 3,
        "kernel8": 3,
        "kernel9": 3,
        "kernel10": 1
    },
    "pooling": {
        "pool1": 2,
        "pool2": 2,
        "pool3": 2,
        "pool4": 2
    },
    "stride": {
        "stride": 1
    },
    "padding": {
        "padding": 1
    },
    "padding_conv":{
        "padding_conv": 0
    },
    "skip_connection": {
        "1": "16",
        "3": "14",
        "5": "12",
        "7": "10"
    },
    "data_format": {
        "data_format":"channels_last"
    }
}
```

</details>

## Layers dictionary
|Layers code |Layers type|
| :---:   | :---: |
| `dsc1d` | Depth Separable Convolution 1D |
| `dsc2d` | Depth Separable Convolution 2D |
| `dsc1d` | Depth Separable Convolution 1D |
| `cnn1d` | Convolutional layer 1D |
| `cnn2d` | Convolutional layer 2D |
| `lstm` | Long Short-Term Memory layer |
| `gru` | Gatered Recurrent Unit layer  |
| `Avgpool1d` |  Average Pooling 1D  |
| `Avgpool2d` |  Average Pooling 2D  |
| `Maxpool1d` |  Max Pooling 1D   |
| `Maxpool1d` |  Max Pooling 2D   |
| `Flatten` |  Flattening layer  |
| `dense` |  Average Pooling 1D  |
| `dropout` |  Dropout layer  |
| `batchNorm1d` |  Batch Normalization 1D  |
| `batchNorm2d` |  Batch Normalization 2D  |
| `relu` |  Rectified Linear Unit  |
| `prelu` | Parametric REctified Linear Unit |
| `tanh` | Hyperbolic Tangent |
| `softmax` | Softmax |
| `sigmoid` | Sigmoid  |
| `Upsample` | Upsample layers |
| `doubleConv` | 2 x [Convolutional layers 2D, Batch Normalization 2D, ReLU] |
| `ElbowUnet` | 2 x [Convolutional layers 2D, Batch Normalization 2D, ReLU] with out_channels influenced by bilinear of None upsampling |
| `UpDoubleConv` | 2 x [Convolutional layers 2D, Batch Normalization 2D, ReLU] with out_channels influenced by bilinear of None upsampling |
| `UpDoubleConv_out` | UpDoubleConv layers that terminate the architecture |
| `output` | dense layers that terminate the architecture |
