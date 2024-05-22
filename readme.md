
 conda create -n torch_graph python=3.8
  conda create -n torch_graph2_gpu python=3.8

 conda env remove -n torch_graph

 conda env list

 conda activate torch_graph
 conda deactivate

pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-1.12.1+cpu.html
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.12.1+cpu.html

pip install pyg_lib==0.2.0 -f https://data.pyg.org/whl/torch-1.12.1+cpu.html
pip install torch-sparse==0.6.14 -f https://data.pyg.org/whl/torch-1.12.1+cpu.html
pip install torch_scatter==2.0.9 -f https://data.pyg.org/whl/torch-1.12.1+cpu.html
pip install torch_cluster==1.6.0 -f https://data.pyg.org/whl/torch-1.12.1+cpu.html
pip install torch_spline_conv==1.2.1 -f https://data.pyg.org/whl/torch-1.12.1+cpu.html


pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-1.12.0+cu113.html



pip install torch_geometric
conda install pyg -c pyg

 conda install pyg=2.2.0=py37_torch_1.12.0_cpu -c pyg

 docker run -dit --name='torch_graph' 40a029b109c0 /bin/bash
 docker run -dit -v /data/snlp/zhangjl/projects:/data/snlp/zhangjl/projects --name='torch_graph' 40a029b109c0 /bin/bash

 docker exec -it torch_graph /bin/bash

 docker commit -m 'torch graph cpu' -a zhangjl19 29f76d566ec2 torch_graph_cpu:230607


 conda install pytorch==1.12.1 cudatoolkit=11.3 -c pytorch

 pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-1.12.0+cu113.html


 docker run -dit --shm-size 32g -v /data/snlp/zhangjl/projects:/data/snlp/zhangjl/projects --name='torch_graph_cpu' 3f4c615b8fcc /bin/bash

docker exec -it torch_graph_cpu /bin/bash


docker run -dit --cpus=32 --shm-size 32g -v /data/snlp/zhangjl/projects:/data/snlp/zhangjl/projects --name='torch_graph' 40a029b109c0 /bin/bash

docker exec -it torch_graph /bin/bash



pip install torch==1.12.0+cpu -f https://download.pytorch.org/whl/torch_stable.html -i https://pypi.tuna.tsinghua.edu.cn/simple 
python - c "import torch ;print(torch.__version__)"

pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.12.0+cpu.html -i https://pypi.tuna.tsinghua.edu.cn/simple 
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.12.0+cpu.html -i https://pypi.tuna.tsinghua.edu.cn/simple 
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.12.0+cpu.html -i https://pypi.tuna.tsinghua.edu.cn/simple 
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.12.0+cpu.html -i https://pypi.tuna.tsinghua.edu.cn/simple 
pip install torch-geometric -i https://pypi.tuna.tsinghua.edu.cn/simple 


https://pytorch-geometric.com/whl/torch-2.0.1%2Bcu117.html

pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-2.0.1%2Bcu117.html -i https://pypi.tuna.tsinghua.edu.cn/simple 
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-2.0.1%2Bcu117.html -i https://pypi.tuna.tsinghua.edu.cn/simple 
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-2.0.1%2Bcu117.html -i https://pypi.tuna.tsinghua.edu.cn/simple 
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-2.0.1%2Bcu117.html -i https://pypi.tuna.tsinghua.edu.cn/simple 
pip install torch-geometric -i https://pypi.tuna.tsinghua.edu.cn/simple 




 conda create -n torch_graph2_cpu python=3.8
 pip install torch==2.0.1+cpu -f https://download.pytorch.org/whl/torch_stable.html -i https://pypi.tuna.tsinghua.edu.cn/simple 
 pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-2.0.1+cpu.html -i https://pypi.tuna.tsinghua.edu.cn/simple 
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-2.0.1+cpu.html -i https://pypi.tuna.tsinghua.edu.cn/simple 
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-2.0.1+cpu.html -i https://pypi.tuna.tsinghua.edu.cn/simple 
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-2.0.1+cpu.html -i https://pypi.tuna.tsinghua.edu.cn/simple 
pip install torch-geometric -i https://pypi.tuna.tsinghua.edu.cn/simple 