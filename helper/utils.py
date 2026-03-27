import numpy as np
from matplotlib.patches import Wedge
import matplotlib.colors as mcolors
from ternary_diagram.utils import three2two
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.gridspec as gridspec
import glob
import os 
from PIL import Image
from ternary_diagram.utils import three2two
from pybaselines import Baseline
from ternary_diagram import TernaryDiagram as Td
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from dtaidistance import dtw
import matplotlib.animation as animation
import imageio.v2 as imageio
import io
from moviepy import VideoClip, TextClip, CompositeVideoClip,VideoFileClip,ImageClip
#print(ImageClip)
import datetime
import matplotlib.cm as cm
import re




def sind(deg):
    # sine with argument in degrees
    return np.sin(deg * np.pi/180)

def tern2cart(T):
    # convert ternary data to cartesian coordinates
    sT = np.sum(T,axis = 1)
    T = 100 * T / np.tile(sT[:,None],(1,3))

    C = np.zeros((T.shape[0],2))
    C[:,1] = T[:,1]*sind(60)/100
    C[:,0] = T[:,0]/100 + C[:,1]*sind(30)/sind(60)
    return C

def cart2tern(C):
    # Convert Cartesian coordinates back to Ternary
    x, y = C[:, 0], C[:, 1]
    
    # Calculate the second component (b)
    b = (2 * y) / np.sqrt(3)
    
    # Calculate the first component (a)
    a = x - 0.5 * b
    
    # Calculate the third component (c)
    c = 1 - a - b
    
    # Return the ternary coordinates as a 3-column array
    return np.column_stack((a, b, c))*100

def draw_pie(ax, ratios, xy, size=0.05, colors=None):
    if colors is None:
        colors = plt.cm.tab10(np.linspace(0, 1, len(ratios)))
    start = 0
    for ratio, color in zip(ratios, colors):
        angle = 360 * ratio
        wedge = Wedge(center=xy, r=size, theta1=start, theta2=start+angle,
                      facecolor=color, edgecolor='k', linewidth=0.2)
        ax.add_patch(wedge)
        start += angle

def remove_bg(x, y):
    """
    To gently remove background from raw data
    """
    new_y= np.zeros_like(y)
    for i in range(y.shape[0]):
        yy=y[i,:]
        baseline_fitter = Baseline(x)
        bkg_1, params_1 = baseline_fitter.asls(yy, lam=1e5, p=0.02)
        y_corrected = yy - bkg_1
        y_corrected[y_corrected < 0] = 0
        new_y[i,:]= y_corrected
    return new_y

def plot_input(xy:np.array, measured_f: np.array, measured_s: np.array, fp:np.array, sp: np.array, save=False, pie_chart=True): 
    """
    1. Plot points selected
    2. Plot components 
    :xy: 2d coordinates 
    :measured_f: index 
    :measured_s: index 
    :fp: (number of sample, 1) -> resistance !! has been shuffled 
    :sp: (number of sample, number of component) -> abundance !! has been shuffled 
    return matplotlib figure 
    """
    xy_= cart2tern(xy)
    fig, axes = plt.subplots(1, sp.shape[1]+1, dpi=72, facecolor="white", figsize=((sp.shape[1]+1)*2.2, 2.2))
    for i in range(sp.shape[1]):
        td = Td(["Te", "Sb", "Mn"], ax=axes[i])
        td.scatter(vector=xy_[measured_s,:], marker="o", c= sp[:,i] ,s=10)
        axes[i].set_title('Phase Component {}'.format(i), fontsize=10)
    td = Td(["Te", "Sb", "Mn"], ax=axes[sp.shape[1]])
    td.scatter(vector=xy_[measured_f,:], marker="o", c= fp ,s=10)
    axes[sp.shape[1]].set_title('Function', fontsize=10)
    fig.suptitle('Input (Random Sampling) with N={}'.format(len(measured_s)))
    plt.tight_layout()
    plt.show()
    recovered_sp = np.full((xy.shape[0], sp.shape[1]), np.nan)
    recovered_sp[measured_s, :] = sp
    np.set_printoptions(suppress=True, precision=3)
    print(recovered_sp[:10,-2])

    if pie_chart:
        n_components = sp.shape[1]
        fig, ax = plt.subplots(dpi=300, figsize=(4, 3))
        td = Td(["Te", "Sb", "Mn"], ax=ax)
        x, y = three2two(xy_ / 100)
        positions = list(zip(x, y))
        default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        colors = default_colors[:n_components] 
        #colors = plt.cm.tab10(np.linspace(0, 1, n_components))
        abundances = recovered_sp

        for i, pos in enumerate(positions):
            abund = abundances[i, :]
            if np.any(np.isnan(abund)):  # skip if any component is NaN
                continue
            norm_abund = abund / np.sum(abund)
            draw_pie(ax, norm_abund, pos, size=0.02, colors=colors)

        ax.set_aspect('equal')
        ax.axis('off')
        labels = [f"C{i}" for i in range(n_components)]
        legend_elements = [Patch(facecolor=c, edgecolor='k', label=l) for c, l in zip(colors, labels)]
        l=ax.legend(handles=legend_elements, title="Components", loc="upper right", fontsize=5)
        l.get_frame().set_edgecolor('black')
        plt.tight_layout()
        plt.show()

def plot_output(pred_mean: np.array, pred_std: np.array, xy:np.array, fp:np.array, sp: np.array, pie_chart=False, save=False): 
    """
    1. Plot points selected
    2. Plot components 
    :pred_mean: mean function of LCM: (number of all sample, number of component+ number of fp)
    :pred_std: mean function of LCM: (number of all sample, number of component+ number of fp)
    :xy: 2d coordinates 
    :measured_f: index 
    :measured_s: index 
    :fp: (number of sample, number of fp) -> resistance 
    :sp: (number of sample, number of component) -> abundance
    return matplotlib figure 
    """
    xy_= cart2tern(xy)
    fig, axes = plt.subplots(1, sp.shape[1]+1, dpi=72, facecolor="white", figsize=((sp.shape[1]+1)*2.2, 2.2))
    for i in range(sp.shape[1]):
        td = Td(["Te", "Sb", "Mn"], ax=axes[i])
        td.scatter(vector=xy_, marker="o", c= pred_mean[:,i] ,s=10)
        axes[i].set_title('Phase Component {}'.format(i), fontsize=10)
    td = Td(["Te", "Sb", "Mn"], ax=axes[sp.shape[1]])
    td.scatter(vector=xy_, marker="o", c=pred_mean[:,sp.shape[1]],s=10)
    axes[sp.shape[1]].set_title('Function', fontsize=10)
    fig.suptitle('Prediction Mean')
    plt.tight_layout()
    plt.show()
    n_components=sp.shape[1]
    abundances=pred_mean[:,:n_components]
    np.set_printoptions(suppress=True, precision=3) 
    print(abundances[:10,-2])
    if pie_chart:
        fig, ax = plt.subplots(dpi=300, figsize=(4, 3))
        td = Td(["Te", "Sb", "Mn"], ax=ax)
        x, y = three2two(xy_/100)
        positions = list(zip(x, y))
        default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        colors = default_colors[:n_components] 
        for xy, pos  in enumerate(positions):
            abund= abundances[xy,:]
            norm_abund= np.array(abund) / sum(abund) 
            draw_pie(ax, norm_abund, pos, size=0.015, colors=colors)
        ax.set_aspect('equal')
        ax.axis('off')
        labels = [f"C{i}" for i in range(n_components)]
        legend_elements = [Patch(facecolor=c, edgecolor='k', label=l) for c, l in zip(colors, labels)]
        l=ax.legend(handles=legend_elements, title="Components", loc="upper right", fontsize=5)
        l.get_frame().set_edgecolor('black')
        plt.tight_layout()
        plt.show()

def reconstruct_xrd(pred_mean: np.array, basis_matrix: np.array, xrd_raw: np.array, xrd_theta: np.array, to_plot=0)-> np.array:
    """
    Reconstruct XRD predicted by abundance
    :pred_mean: mean function of LCM: (number of all sample, number of component) (177, 7)
    :basis_matrix: the basic components (7,901)
    :xrd_raw: raw (true) data to be compared  
    :xrd_theta: theta for plot (901, )
    :to_plot: index to plot and represent
    return similarity score 
    """
    reconstructed_xrd= np.dot(pred_mean, basis_matrix)
    # cosine similarity
    similarity = cosine_similarity(reconstructed_xrd, xrd_raw)
    similarity_score= similarity[0][0]
    # dynamic time warpping 
    similarity_matrix = np.zeros((xrd_raw.shape[0],))
    for i in range(xrd_raw.shape[0]):
        dist = dtw.distance(reconstructed_xrd[i,:], xrd_raw[i,:], window=20)
        #similarity_dtw = 1 / (1 + dist)
        similarity_matrix[i]= dist
    dtw_score= np.mean(similarity_matrix)
    fig, axes = plt.subplots(1, 2, dpi=72, figsize=(12, 3))
    axes[0].plot(xrd_theta,xrd_raw[to_plot, :], c='black',label='raw')
    axes[0].plot(xrd_theta,reconstructed_xrd[to_plot, :], c='red', label='reconstructed')
    axes[0].set_title('DTW distance: {:.2f}'.format(dtw_score))
    axes[0].set_xlabel(r"$2\theta$ (°)", fontsize=10)
    axes[0].set_ylabel("Intensity (a.u.)", fontsize=10)
    l=axes[0].legend()
    l.get_frame().set_edgecolor('white')
    #axes[0].set_title('Reconstruction Example {}'.format(to_plot))
    im = axes[1].imshow(similarity, cmap='viridis', vmin=0, vmax=1)
    axes[1].invert_yaxis()
    axes[1].set_title('Cosine Similarity Matrix: {:.2f}'.format(similarity_score))
    axes[1].set_xlabel('Raw Sample Index')
    axes[1].set_ylabel('Reconstructed Sample Index')
    #plt.tight_layout()
    plt.show()

    return reconstructed_xrd
    
    
def plot_gif(FOLDER_PATH, name='animation', frame_max=None, duration=500):

    # 1. Glob all PNGs (sorted by name)
    image_files = sorted(
        [f for f in glob.glob(FOLDER_PATH + '*.png') if '_live' not in os.path.basename(f)],
        key=lambda x: int(os.path.basename(x).split('.')[0])
    )

    # 2. Load images
    if frame_max is None:
        frames = [Image.open(img) for img in image_files]
    else:
        image_files = image_files[:frame_max]
        frames = [Image.open(img) for img in image_files]

    # 3. Save as GIF
    frames[0].save(
        FOLDER_PATH+'{}.gif'.format(name),
        format='GIF',
        append_images=frames[1:],  # all other frames
        save_all=True,
        duration=duration,  # time per frame in ms
        loop=0  # loop forever
    )
def plot_gif_live(FOLDER_PATH, name='animation_live', frame_max=None, duration=500):
    """
    126 seconds=126,000 milliseconds
        126,000 ms
        ÷
        20
         frames
        =
        6
        ,
        300
         ms per frame
        126,000 ms÷20 frames=6,300 ms per frame
    """
    # 1. Glob all PNGs (sorted by name)
    image_files = sorted(
        glob.glob(FOLDER_PATH + '*_live.png'),
        key=lambda x: int(os.path.basename(x).split('_')[0])
    )

    # 2. Load images
    if frame_max is None:
        frames = [Image.open(img) for img in image_files]
    else:
        image_files = image_files[:frame_max]
        frames = [Image.open(img) for img in image_files]


    # 3. Save as GIF
    frames[0].save(
        FOLDER_PATH+'{}.gif'.format(name),
        format='GIF',
        append_images=frames[1:],  # all other frames
        save_all=True,
        duration=duration,  # time per frame in ms
        loop=0  # loop forever
    )
def calculate_dtw(reconstructed_xrd: np.array, xrd_raw: np.array)->float:
    """
    calculate dynamic time warping distance
    :reconstructed_xrd: array a
    :xrd_raw: array b
    return the mean of all the points
    """
    similarity_matrix = np.zeros((xrd_raw.shape[0],))
    for i in range(xrd_raw.shape[0]):
        dist = dtw.distance(reconstructed_xrd[i,:], xrd_raw[i,:], window=20)
        similarity_matrix[i]= dist
    dtw_score= np.mean(similarity_matrix)
    return dtw_score 
def draw_piechart(ax, xy: np.array, n_components: int, abundances: np.array, circle=False, sequence=None, init=None):
    """
    Help draw pie chart 
    :ax: the ax to plot 
    :xy: 3D cooredinates 
    :n_components: number of phase decomposition
    :abundances: the phase composition 
    :circle: to red circle the last point 
    :sequence: the number of iteration to plot arrow for decision path -> init 
    :int: the number of initital points -> int 
    :arrow_path: this is for path finding 
    return figure 
    """
    td = Td(["Te", "Sb", "Mn"], ax=ax)
    if circle:
        td.scatter(vector=xy[-1,:].reshape(1, -1), facecolors='none', edgecolors='red', linewidths=1.5, s=150)
    x, y = three2two(xy/100)
    positions = list(zip(x, y))
    #default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    #colors = default_colors[:n_components] 
    cmap= cm.get_cmap("viridis", n_components) 
    colors = [cmap(i) for i in range(n_components)]
    if sequence is None:
        for xyxy, pos  in enumerate(positions):
            abund= abundances[xyxy,:]
            norm_abund= np.array(abund) / sum(abund) 
            draw_pie(ax, norm_abund, pos, size=0.03, colors=colors)
    else:
        positions_crop= positions[:int(init+sequence)+1]
        for xyxy, pos  in enumerate(positions_crop):
            abund= abundances[xyxy,:]
            norm_abund= np.array(abund) / sum(abund) 
            draw_pie(ax, norm_abund, pos, size=0.03, colors=colors)
        start = positions_crop[-2]
        end = positions_crop[-1]
        ax.annotate('', xy=end, xytext=start, arrowprops=dict(arrowstyle='->', color='black', lw=1))


    ax.set_aspect('equal')
    ax.axis('off')
    labels = [f"C{i}" for i in range(n_components)]
    legend_elements = [Patch(facecolor=c, edgecolor='k', label=l) for c, l in zip(colors, labels)]
    l=ax.legend(handles=legend_elements, loc="upper left", fontsize=6,bbox_to_anchor=(-0.22, 0.9))
    l.get_frame().set_edgecolor('white')
    """
 
    if sequence is not None:
        for i, measured in enumerate(sequence[:-init_n]):
            start = positions[i+init_n-1]
            end = positions[i+init_n]
            ax.annotate(
                '', xy=end, xytext=start,
                arrowprops=dict(arrowstyle='->', color='black', lw=1))
    """
            
    return ax 

def gp_model(model_dict, f= 'coreg matrix'):
    """
    :model_dict: gp_dict
    Help extract B for current iteration
    """
    W= model_dict['kernel']['parts'][1]['W']
    w= np.array(W, dtype=float)
    K= model_dict['kernel']['parts'][1]['kappa']
    k= np.array(K, dtype=float)
    B = w @ w.T + np.diag(k)

    if f=='coreg matrix':
        ans= B
    elif f== 'correlation':
        D = np.sqrt(np.diag(B))
        ans = B / np.outer(D, D)
    elif f== 'rank':
        ans= w 
    return ans 

def plot_decision_path(FOLDER_PATH):
    """
    :FOLDER_PATH: dictionary 
    Plot decision path made by the algorithm
    Both pm and fm are plotted 
    """
    file_path = FOLDER_PATH + 'data.pkl'
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
    else:
        print(f"Data File does not exist: {file_path}")
    file_path = FOLDER_PATH + 'log.pkl'
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            log = pickle.load(f)
    else:
        print(f"Log File does not exist: {file_path}")

    xy= cart2tern(data['XY'])
    # Save GIF pm 
    measured_s= log['measured_s']
    n_iteration= len(log['log'])
    sp= log['log'][int(n_iteration-1)]['sp_input']
    init= data['INIT']
    frames=[]
    total_iteration= data['LOOP']
    for it in range(total_iteration):
        fig, ax = plt.subplots(1, 1, dpi=200, facecolor="white", figsize=((3,3)))
        ax.clear()
        draw_piechart(ax, xy[measured_s[:-1],:], sp.shape[1], sp, sequence=it, init=init)
        buf = io.BytesIO()
        fig.savefig(buf, format="png")   # render to buffer
        buf.seek(0)
        frames.append(imageio.imread(buf))
        plt.tight_layout()
        plt.close(fig)
    name= 'decision_path_pm'
    duration=400
    imageio.mimsave(FOLDER_PATH+f"{name}.gif", frames, duration=duration, loop=0)

    # Save GIF fm 
    measured_f= log['measured_f']
    n_iteration= len(log['log'])
    fp= data['known_f']
    init= data['INIT']
    frames=[]
    total_iteration= data['LOOP']
    for it in range(total_iteration):
        fig, ax = plt.subplots(1, 1, dpi=200, facecolor="white", figsize=((3,3)))
        ax.plot(xy[measured_f[:-1],:])
        buf = io.BytesIO()
        fig.savefig(buf, format="png")   # render to buffer
        buf.seek(0)
        frames.append(imageio.imread(buf))
        plt.tight_layout()
        plt.close(fig)
    name= 'decision_path_fm'
    duration=400
    imageio.mimsave(FOLDER_PATH+f"{name}.gif", frames, duration=duration, loop=0)


def make_video(FOLDER_PATH):
    ## needs to be fixed 
    clip = (VideoFileClip("/Users/chihyulee/work/multi-instr/HetGP/AL_21_2min_title.mp4"))
    timelapse_duration = clip.duration

    # Create a running timestamp overlay
    def make_time_frame(t):
        ts = str(datetime.timedelta(seconds=int(t * 150)))
        txt = TextClip(font="Arial.ttf", text=ts, font_size=50, color='darkred', bg_color='white').with_position(('center', 'bottom'))
        return txt.get_frame(0)

    txt_clip = VideoClip(make_time_frame, duration=timelapse_duration)

    final_video = CompositeVideoClip([clip, txt_clip])
    final_video.write_videofile("demo_final.mp4")

def make_video_new():
    clip = VideoFileClip("/Users/chihyulee/work/multi-instr/HetGP/AL_21_2min_title.mp4")
    timelapse_factor = 150
    timelapse_duration = clip.duration

    def make_time_frame(t):
        total_seconds = int(t * timelapse_factor)
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        ts = f"{hours}h {minutes}m {seconds}s"
        txt = TextClip(font="Arial.ttf", text=ts, font_size=50, color='darkred', bg_color='white').with_position(('center', 'bottom'))
        return txt.get_frame(0)

    txt_clip = VideoClip(make_time_frame, duration=timelapse_duration)

    final_video = CompositeVideoClip([clip, txt_clip])
    final_video.write_videofile("demo_final_IT.mp4")
    

def make_video_image(video_path, images_folder, ending=21):
    """
    function to integrate images with video 
    """
    def sort_key_p(f):
        basename = os.path.basename(f)
        match = re.match(r"(\d+)_pm\.png", basename)
        return int(match.group(1)) if match else 0
    def sort_key_f(f):
        basename = os.path.basename(f)
        match = re.match(r"(\d+)_fm\.png", basename)
        return int(match.group(1)) if match else 0
    # images 
    image_files_p = glob.glob(os.path.join(images_folder, "*_pm.png"))
    image_files_pm = sorted(image_files_p, key=sort_key_p)[:ending]
    image_files_f = glob.glob(os.path.join(images_folder, "*_fm.png"))
    image_files_fm = sorted(image_files_f, key=sort_key_f)[:ending]

    # video
    clip = VideoFileClip(video_path)
    fps = clip.fps
    interval = clip.duration/(ending-1)
    img_w= 300
    
    image_clips_pm = []
    for i, img_path in enumerate(image_files_pm):
        img_clip = ImageClip(img_path) \
        .with_duration(interval)\
        .with_start((i+1) * interval) \
        .with_position((clip.w/2 - img_w, 0))\
        .resized(width=img_w, height= img_w)
        image_clips_pm.append(img_clip)

    image_clips_fm = []
    for i, img_path in enumerate(image_files_fm):
        img_clip = ImageClip(img_path) \
        .with_duration(interval)\
        .with_start((i+1) * interval) \
        .with_position(('right', 'top'))\
        .resized(width=img_w, height= img_w)
        image_clips_fm.append(img_clip)\

    # Combine everything
    final = CompositeVideoClip([clip, *image_clips_pm,*image_clips_fm ])
    final= final.subclipped(0, 124)

    # Write output
    final.write_videofile("demo_final_IT_new_03232026.mp4", fps=clip.fps)

def snapshot(video_path, t):
    clip = VideoFileClip(video_path)
    clip.save_frame("snapshot_03232026_{}.png".format(t), t=t)

 
    




    

    


    








    



