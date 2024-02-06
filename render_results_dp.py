import pickle
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.tri as tri
import cv2
import PIL.Image as Image
from tqdm import tqdm
import datetime
import glob
import os
import plotly.graph_objects as go
import plotly.io as pio



def fig2data(fig):
    """
    fig = plt.figure()
    image = fig2data(fig)
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    
    # draw the renderer
    fig.canvas.draw()
 
    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)
 
    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)
    image = Image.frombytes("RGBA", (w, h), buf.tobytes())
    image = np.asarray(image)
    return image

def plot3D_position_stress(data, n, step):

        # Set your desired axis limits
        x_limit = [-0.1, 0.3]
        y_limit = [-0.1, 0.5]
        z_limit = [-0.1, 0.3]
        # Sample data for demonstration
        
        positions = data[:,:-1]
        node_type = n.reshape(-1)
        value = data[:, -1]
        # Node type color map
        cmap = {1: 'blue', 3: 'black'}
        # Create a 3D scatter plot for nodes with circles and contours

        type_trace_list = []
        for node_type_value in np.unique(node_type).tolist():
            if node_type_value == 0:
                type_trace = go.Scatter3d(
                    x=positions[:, 0],
                    y=positions[:, 1],
                    z=positions[:, 2],
                    mode='markers',
                    marker=dict(
                        size=5,
                        color=value,
                        colorscale='YlOrRd',  # You can choose a different colorscale
                        cmin=0,
                        cmax=120000,
                        colorbar=dict(title='S.Mises'),
                    ),
                    hoverinfo='text',
                )
            else:
                indices = (node_type == node_type_value)
                type_trace = go.Scatter3d(
                    x=positions[indices, 0],
                    y=positions[indices, 1],
                    z=positions[indices, 2],
                    mode='markers',
                    marker=dict(
                        size=5,  # Adjust the size of the circle
                        color=cmap[node_type_value],  # Transparent fill
                    ),
                )
            type_trace_list.append(type_trace)


        layout = go.Layout(
            scene=dict(
                xaxis=dict(title='X', range=x_limit),
                yaxis=dict(title='Y', range=y_limit),
                zaxis=dict(title='Z', range=z_limit),
                aspectmode='cube',
                camera=dict(eye=dict(x=1.2, y=1.2, z=0.75)),
                ),
                title='Prediction\nTime @ %.2f s'%(step*0.01)
        )


        # Create the figure
        fig = go.Figure(data=type_trace_list, layout=layout)

        return fig


result_files = glob.glob('result/*.pkl')
os.makedirs('videos', exist_ok=True)

for index, file in enumerate(result_files):

    with open(file, 'rb') as f:
        result, n = pickle.load(f)
    n = n.cpu().detach().numpy()

    file_name = 'videos/output%d.mp4'%index

    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 保存视频的编码
    out = cv2.VideoWriter(file_name, fourcc, 20.0, (1700,800))

    r_t = result[0][:, 0]

    v_max = np.max(r_t)
    v_min = np.min(r_t)

    colorbar = None
    skip=5
    
    def render(i):

        step = i * skip
        target = result[1][step]
        predicted = result[0][step]

        fig = plot3D_position_stress(predicted, n, step)

        fig.write_image(f'./frame.png', width=1000 * 2, height=800 * 2, scale=2)

        img = cv2.imread(f'./frame.png')
        out.write(img)


    for i in tqdm(range(399), total=400//skip):
        if i*skip < 350:
            render(i)
    out.release()
    print('video %s saved'%file_name)