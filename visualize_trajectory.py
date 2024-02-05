from dataset import FPCdp, FPCdp_ROLLOUT

from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T
from train_dp import FaceToEdgeTethra

from concurrent.futures import ThreadPoolExecutor

from pathlib import Path
import os
from tqdm import tqdm
import cv2
from copy import deepcopy
import plotly.graph_objects as go
from plotly.offline import plot

import matplotlib.pyplot as plt

import torch


dataset_dir = "./data/deforming_plate"
batch_size = 1
noise_std = 2e-2

print_batch = 10
save_batch = 200


def make_plot_plotly3D(x, n , snapshot, path=None, edge_index=None):
    # Set your desired axis limits
    x_limit = [-0.1, 0.3]
    y_limit = [-0.1, 0.5]
    z_limit = [-0.1, 0.3]
    # Sample data for demonstration
    positions = x[:,:-1]
    node_type = n[:, 0]
    value = x[:, -1]
    label_node = torch.arange(0, len(node_type))
    # Node type color map
    cmap = {1: 'blue', 3: 'black'}
    # Create a 3D scatter plot for nodes with circles and contours

    type_trace_list = []
    for node_type_value in torch.unique(node_type).tolist():
        if node_type_value == 0:
            type_trace = go.Scatter3d(
                x=positions[:, 0],
                y=positions[:, 1],
                z=positions[:, 2],
                mode='markers+text',
                text=label_node,
                marker=dict(
                    size=5,
                    color=value,
                    colorscale='YlOrRd',  # You can choose a different colorscale
                    cmin=0,
                    cmax=120000,
                    colorbar=dict(title='S.Mises'),
                ),
                hoverinfo='text',
                #text=[f'S.Mises: {val:.2f}' for val in value]
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

    if edge_index is not None:
        source = edge_index[0]
        target = edge_index[1]
        # # TESTING
        # at = torch.argwhere(node_type == 1).squeeze()
        # for i in at:
        #     index_i = torch.argwhere(source == i)
        #     wrong_edge = torch.argwhere(target[index_i] > torch.max(at))
        #     if len(wrong_edge) > 0:
        #         print(target[index_i])
        #         raise ValueError
        # print('Now object!')
        # at = torch.argwhere(node_type != 1).squeeze()
        # negative = torch.argwhere(positions[at,:] < -0.1)
        # if len(negative) > 0:
        #     print('Wrong')
        # for i in at:
        #     index_i = torch.argwhere(source == i)
        #     wrong_edge = torch.argwhere(target[index_i] <= torch.min(at))
        #     if len(wrong_edge) > 0:
        #         target[index_i]
        #         raise ValueError
            
        # Extract coordinates for source and target nodes
        source_coordinates = [positions[i] for i in source]
        target_coordinates = [positions[i] for i in target]
        # Flatten the coordinate lists for Scatter3d
        x = [coord[0] for sublist in zip(source_coordinates, target_coordinates) for coord in sublist]
        y = [coord[1] for sublist in zip(source_coordinates, target_coordinates) for coord in sublist]
        z = [coord[2] for sublist in zip(source_coordinates, target_coordinates) for coord in sublist]

        # Create scatter plot for edges
        scatter_edges = go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode='lines',
            line=dict(color='gray', width=1),
            )
        type_trace_list.append(scatter_edges)

    layout = go.Layout(
        scene=dict(
            xaxis=dict(title='X', range=x_limit),
            yaxis=dict(title='Y', range=y_limit),
            zaxis=dict(title='Z', range=z_limit),
            aspectmode='cube',
            camera=dict(eye=dict(x=1.2, y=1.2, z=0.75)),
            )
    )
    # Create the figure
    fig = go.Figure(data=type_trace_list, layout=layout)

    if path is not None:
        # Save the figure to a file (replace 'trajectory_snapshot.png' with your desired file name and format)
        fig.write_image(path + f'/frame{snapshot:03d}.png', width=1000 * 2, height=800 * 2, scale=2)
    else:
        plot(fig)


def make_video(image_folder):
    # Output video file (change the extension to '.mp4' for MP4 format)
    video_name = image_folder + '/trajectory.mp4'
    # Get the list of image files in the directory
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    # Sort the images based on their filenames
    images.sort()
    # Set the frame width and height (adjust as needed)
    frame_width = 1800
    frame_height = 1500
    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video = cv2.VideoWriter(video_name, fourcc, 10, (frame_width, frame_height))
    # Iterate over the images and write each frame to the video
    for image in images:
        img_path = os.path.join(image_folder, image)
        frame = cv2.imread(img_path)
        # Resize the image to match the frame dimensions
        frame = cv2.resize(frame, (frame_width, frame_height))
        # Write the frame to the video
        video.write(frame)
    # Release the video writer and close the OpenCV window
    video.release()
    cv2.destroyAllWindows()


def process_trajectory(i, dataset):

    path = Path(f'./outputs/trajectory{i}')
    path.mkdir(exist_ok=True, parents=True)
    dataset.change_file(i)
    frame = -1

    for graph in tqdm(DataLoader(dataset=dataset, batch_size=batch_size, num_workers=1)):
        frame += 1
        if frame % 5 != 0:
             continue
        graph = FaceToEdgeTethra().forward(graph)
        #path = str(path),

        make_plot_plotly3D(graph.x, graph.n, frame, path=None, edge_index=graph.edge_index)
        break

    make_video(str(path))


if __name__ == '__main__':

    transformer = T.Compose([T.Cartesian(norm=False), T.Distance(norm=False)])

    # # # Set dataset
    # path = Path(f'./outputs/trajectory3')
    # dataset = FPCdp_ROLLOUT(dataset_dir=dataset_dir, split='test')
    # idx = 0
    # dataset.change_file(idx)
    # loader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=1, shuffle=False)
    # for graph in loader:
    #     graph = FaceToEdgeTethra().forward(graph)
    #     graph = transformer(graph)
    #     make_plot_plotly3D(graph.x, graph.n , 0, edge_index=graph.edge_index)
    #     break


    ####################################
    ####################################
    ####################################
    # # UNCOMMENT TO VISUALIZE TRAJECTORY
    # Set dataset
    dataset_fpc = FPCdp_ROLLOUT(dataset_dir=dataset_dir, split='test')
    # Number of threads
    num_threads = 8
    # Your original range
    trajectory_range = [3] #range(0, 100, 10)
    parameters_threads = [deepcopy(dataset_fpc) for _ in range(len(trajectory_range))]
    # Set threads
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        executor.map(process_trajectory, trajectory_range, parameters_threads)


