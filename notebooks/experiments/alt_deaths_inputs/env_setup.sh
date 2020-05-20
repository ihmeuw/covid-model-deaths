pip install jupyter jupyterlab scipy  numpy pandas xlrd openpyxl pyarrow tables \
statsmodels dill statsmodels matplotlib seaborn pyyaml pytest xspline db_queries PyPDF2 && \
conda install -c conda-forge cyipopt && \
git clone https://github.com/zhengp0/limetr.git && \
cd limetr && make install && cd .. && \
git clone https://github.com/ihmeuw-msca/MRTool.git && \
cd MRTool && git checkout seiir_model && python setup.py install && cd .. && \
git clone https://github.com/zhengp0/SLIME.git && \
cd SLIME && python setup.py install && \
cd .. && rm -rf limetr && rm -rf MRTool && rm -rf SLIME
