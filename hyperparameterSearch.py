from mof import MOF_CGCNN
import csv,os,datetime,json
import numpy as np
from ax import ParameterType, ChoiceParameter, RangeParameter,  SearchSpace, SimpleExperiment, modelbridge
from plotly.offline import plot
from ax.plot.contour import interact_contour,plot_contour
from ax.plot.slice import plot_slice
import matplotlib.pyplot as plt
from ax.plot.scatter import interact_fitted
from sklearn.model_selection import train_test_split



with open('./traning_val.csv') as f:
    readerv = csv.reader(f)
    trainandval = [row for row in readerv]
with open('./test.csv') as f:
    readerv = csv.reader(f)
    test = [row for row in readerv]
from sklearn.model_selection import train_test_split
train, val = train_test_split(trainandval, test_size=0.11, random_state=24)
#file path
root = './cif'



def train_evaluate(parameterization):
    mof = MOF_CGCNN(cuda=True, root_file=root,trainset = train, valset=val,testset=test,epoch = 500,lr=0.002,optim='Adam',batch_size=24,h_fea_len=parameterization.get('h_fea_len'),n_conv=parameterization.get('conv_layer'),lr_milestones=[150],weight_decay=parameterization.get('decay'),dropout=parameterization.get('dropout'))
    mae = mof.train_MOF()
    return mae.item()




search_space = SearchSpace([RangeParameter(name='h_fea_len', parameter_type=ParameterType.INT,lower=100, upper=600, log_scale=False),
		RangeParameter(name='conv_layer',parameter_type=ParameterType.INT,lower=4, upper=6, log_scale=False),
		RangeParameter(name='dropout', parameter_type=ParameterType.FLOAT,lower=0.1, upper=0.6, log_scale=False),
		RangeParameter(name='decay', parameter_type=ParameterType.FLOAT,lower=1e-10, upper=1e-4, log_scale=True)])

exp = SimpleExperiment(
    name='hypar_MOF_CGCNN_',
    search_space=search_space,
    evaluation_function=train_evaluate,
    objective_name='mae',
    minimize=True  # True means minimize, False means maximize
)
sobol = modelbridge.get_sobol(search_space=exp.search_space)
print(f"\nRunning Sobol initialization trials...\n{'=' * 40}\n")
for _ in range(10):
    exp.new_trial(generator_run=sobol.gen(1))

for i in range(30):
    print(f"\nRunning GP+EI optimization trial {i + 1}/{5}...\n{'=' * 40}\n")
    gpei = modelbridge.get_GPEI(experiment=exp, data=exp.eval())
    exp.new_trial(generator_run=gpei.gen(1))
output_dir = os.path.join('./Ax_output', 'cgcnn', datetime.datetime.now().strftime('%m%d-%H%M%S'))
os.makedirs(output_dir)
df = exp.eval().df
df.to_csv(os.path.join(output_dir, 'exp_eval.csv'), index=False)
best_arm_name = df.arm_name[df['mean'] == df['mean'].max()].values[0]
exp_arm = {k: v.parameters for k, v in exp.arms_by_name.items()}
exp_arm['best'] = best_arm_name
print('Best arm:\n', str(exp.arms_by_name[best_arm_name]))
with open(os.path.join(output_dir, 'exp_arm.json'), 'w') as f:
    json.dump(exp_arm, f)
# Contour Plot
os.makedirs(os.path.join(output_dir, 'contour_plot'))
for metric in ['mae']:
    contour_plot = interact_contour(model=gpei, metric_name=metric)
    plot(contour_plot.data, filename=os.path.join(output_dir, 'contour_plot', '{}.html'.format(metric)))
# Slice Plot
os.makedirs(os.path.join(output_dir, 'slice_plot'))
for param in ["h_fea_len","conv_layer","dropout","decay"]:
    slice_plot = plot_slice(gpei, param, "mae")
    plot(slice_plot.data, filename=os.path.join(output_dir, 'slice_plot', '{}.html'.format(param)))
# Tile Plot
tile_plot = interact_fitted(gpei, rel=False)
plot(tile_plot.data, filename=os.path.join(output_dir, 'tile_plot.html'))
