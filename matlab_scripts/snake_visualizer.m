
clear; close all;
rng('default');

n_links = 16; % The robot has 16 joints and 16 links.
link_length = 5.105; % module length (cm)


waypoints = { 
        [30,-30,30]',

    };
T_base_initial = eye(4);

load('my_data_3.mat')
q_initial=my_matrix(end,:);
% 'T_base_initial' is the starting pose of the HEAD at t=0.


safety_margin = 0.01; 
snake_thickness = 3;
enable_collision_avoidance = true;
actual_obs=[
            22-snake_thickness -35 27 22+snake_thickness -10 37;
            36-snake_thickness -35 27 36+snake_thickness -10 37;
            24.5 -35 26 35.5 -10 27; 
            24.5 -35 37 35.5 -10 38; 
            20 -38 20 40 -35 45;
            ];



% =========================================================================
% 2. PREVIEW INITIAL SETUP
% =========================================================================
fprintf('Displaying initial problem setup for sanity check...\n');
fig_preview = figure('Name', 'Optimization Setup Preview', 'NumberTitle', 'off');
set(fig_preview, 'WindowState', 'maximized');
ax_preview = axes(fig_preview); grid(ax_preview, 'on'); axis(ax_preview, 'equal'); hold(ax_preview, 'on');
max_range=0; for yy = 1:length(waypoints), plot3(ax_preview, waypoints{yy}(1), waypoints{yy}(2), waypoints{yy}(3), 'r*', 'MarkerSize', 15, 'LineWidth', 2, 'DisplayName', sprintf('Waypoint %d',yy)); end
% for k = 1:length(dummy_cuboids), plot_cuboid_faces(ax_preview, dummy_cuboids{k}.dims, dummy_cuboids{k}.axis); end
if exist('actual_obs', 'var') && ~isempty(actual_obs)
    num_obs = height(actual_obs);
    for k = 1:num_obs
        % Get the coordinates for the k-th obstacle
        obs_dims = actual_obs(k,:);
        % obs_dims = acutal_obs( (k-1)*6 + 1 : k*6 );
        
        % Plot it using the new function
        plot_solid_cuboid(ax_preview, obs_dims, 'r', 0.2); % Red, 20% opaque
    end
end
% Call the generalized FK, specifying 'HEAD_BASE' for the initial pose.

[P_initial, ~] = forwardKinematics(q_initial', link_length, T_base_initial, 'HEAD_BASE');
plot3(ax_preview, P_initial(1,:), P_initial(2,:), P_initial(3,:), 'o-', 'LineWidth', 2, 'MarkerSize', 4, 'Color', [0.1, 0.4, 0.8], 'DisplayName', 'Snake Robot (t=0)');
% Auto-scaling logic
for yy=1:length(waypoints),max_range=max([max_range,abs(waypoints{yy}(:))']);end; max_range=max([max_range,abs(P_initial(:)')]);
lim = ceil(max_range*1.1); if lim==0, lim=5; end; axis(ax_preview,[-lim lim -lim lim -lim lim]); view(ax_preview, 3);
xlabel(ax_preview,'X');ylabel(ax_preview,'Y');zlabel(ax_preview,'Z');title(ax_preview,'Initial Setup. Press any key to start.'); legend(ax_preview,'show','Location','northwest');
drawnow; disp('Sanity Check: Press any key to start optimization...'); 
% pause;
close(fig_preview); fprintf('Preview confirmed. Starting optimization...\n');
tic;


% =========================================================================
% 3. SET UP AND SOLVE THE OPTIMIZATION PROBLEM FOR EACH SEGMENT
% =========================================================================

% --- Initialize Full Trajectory Storage ---
full_q_dot_opt = zeros(n_links, 1); 
full_u_opt = zeros(n_links, 1); 

% --- Store poses for visualization ---
full_T_base_trajectory = {};
full_base_end_trajectory = {};
segment_end_times = [0]; % Stores the cumulative end time of each segment



% --- Initialize Full Trajectory Storage ---
smooth_full_q_opt = q_initial'; 
smooth_full_time_vector = 0;

% --- Store poses for visualization ---
smooth_full_T_base_trajectory = {};
smooth_full_base_end_trajectory = {};
smooth_segment_end_times = [0]; 


% --- Initialize State for the First Segment ---
current_x_initial = [q_initial, zeros(1, n_links)]';
T_base_current = T_base_initial;


n_states = 2 * n_links;
n_controls = n_links;
    
curr_path = [];
full_smooth_path=[];
locs=[];
spath=[];

full_T_base_trajectory{1} = T_base_current; 
full_base_end_trajectory{1} = 'HEAD_BASE';
segment_end_times = [segment_end_times, 5];
full_time_vector=linspace(0, 20, height(my_matrix));
full_q_opt = my_matrix';
output_filename = 'snake_traj.mat';
save(output_filename, 'full_q_opt', 'full_time_vector', ...
    'link_length', 'n_links', 'waypoints', ...
    'full_T_base_trajectory', 'full_base_end_trajectory', 'segment_end_times', ...
    'actual_obs'); % NEW: Save directions
fprintf('\nFull trajectory data saved to %s\n', output_filename);
fprintf('All tasks complete.\n'); beep;
toc


% =========================================================================
% LOCAL FUNCTIONS
% =========================================================================
function [smoothed_path, skipped_counts] = smooth_path_shortcut(path, link_length, T_base, base_end, obstacles, numobs, max_iterations)
    % Initialize
    smoothed_path = path;
    
    % Track original indices. Initially, it points to every row in the original path.
    original_indices = 1:size(path, 1);
    
    if size(smoothed_path, 1) < 3
        skipped_counts = zeros(1, size(smoothed_path, 1) - 1);
        return;
    end
    
    n_links = size(path, 2);
    
    for k = 1:max_iterations
        % 1. Pick two random indices (i and j) based on current smoothed_path size
        len = size(smoothed_path, 1);
        if len < 4, break; end

        idx1 = randi(len - 3);
        idx2 = randi([idx1 + 2, len-1]); % Ensure idx2 is at least 2 steps ahead
        
        q1 = smoothed_path(idx1, :);
        q2 = smoothed_path(idx2, :);
        
        % 2. Check if straight line connection is valid
        % We check discrete points along the line segment
        valid_shortcut = true;
        steps = 5000; % Check 10 points along the shortcut
        for t = linspace(0, 1, steps)
            q_check = (1-t)*q1 + t*q2;
            % Note: q_check is 1xN, function likely expects Nx1, so we transpose
            if ~is_collision_free_RRT(q_check', link_length, T_base, base_end, obstacles, numobs)
                valid_shortcut = false;
                break;
            end
        end
        
        % 3. If valid, remove intermediate nodes
        if valid_shortcut
            % Replace path with [start ... q1, q2 ... end]
            smoothed_path = [smoothed_path(1:idx1, :); smoothed_path(idx2:end, :)];
            
            % CRITICAL STEP: Apply the exact same deletion to the index tracker
            original_indices = [original_indices(1:idx1), original_indices(idx2:end)];
        end
    end
    
    % 4. Calculate skipped points vector
    % diff(original_indices) gives the gap between connected indices.
    % Subtract 1 to get the number of points *between* them.
    % Example: if indices are [1, 5], diff is 4. Points skipped are indices 2,3,4 (count=3).
    skipped_counts = diff(original_indices) - 1;
end



% --- Generalized Forward Kinematics (v13+) ---
function [P_world, T_ee_world] = forwardKinematics(q, link_length, T_base, base_end)
    n_links = length(q); 
    T_links_in_head_frame=cell(n_links+1,1); T_links_in_head_frame{1}=eye(4); T_current=eye(4);
    for i=1:n_links
        if mod(i,2)==1,T_joint=Rz(q(i));else,T_joint=Ry(q(i));end % Z-Y-Z-Y...
        T_current=T_current*T_joint*Tx(link_length); 
        T_links_in_head_frame{i+1}=T_current; 
    end
    T_tail_in_head_frame = T_links_in_head_frame{n_links+1};
    if strcmpi(base_end, 'HEAD_BASE') % 'REACH'
        T_world_to_head = T_base; 
        T_ee_world = T_world_to_head * T_tail_in_head_frame;
    elseif strcmpi(base_end, 'TAIL_BASE') % 'PULL'
        T_world_to_head = T_base * inv(T_tail_in_head_frame); 
        T_ee_world = T_world_to_head;
    else, error('Invalid base_end specified.'); end
    P_world = zeros(3, n_links + 1);
    for i=1:n_links+1
        T_link_world=T_world_to_head*T_links_in_head_frame{i}; 
        P_world(:,i)=T_link_world(1:3,4); 
    end
end
% --- Helper: Cuboid Violations (Identical to v10.3) ---

% switch upper(p), case 'X',fs=[f_Ym;f_YM;f_Zm;f_ZM];case 'Y',fs=[f_Xm;f_XM;f_Zm;f_ZM];case 'Z',fs=[f_Xm;f_XM;f_Ym;f_YM];otherwise,fs=[f_Ym;f_YM;f_Zm;f_ZM];end; patch(ax,'Vertices',v,'Faces',fs,'FaceColor','b','FaceAlpha',0.1,'EdgeColor','b','LineStyle','--'); 
% --- Helper: Homogeneous Transforms (Identical to v10.3) ---
function T=Rz(t),c=cos(t);s=sin(t);T=[c -s 0 0;s c 0 0;0 0 1 0;0 0 0 1];end 
function T=Ry(t),c=cos(t);s=sin(t);T=[c 0 s 0;0 1 0 0;-s 0 c 0;0 0 0 1];end
function T=Tx(d),T=[1 0 0 d;0 1 0 0;0 0 1 0;0 0 0 1];end

% =========================================================================
% NEW (v16): RRT-Connect Implementation and Helpers
% =========================================================================

function plot_solid_cuboid(ax, c, color, alpha)
% Plots a solid 6-faced cuboid
% c = [xmin, ymin, zmin, xmax, ymax, zmax]
    if isempty(c), return; end
    x_c=[c(1),c(4)]; y_c=[c(2),c(5)]; z_c=[c(3),c(6)];
    
    % Vertices
    v=[x_c(1),y_c(1),z_c(1); x_c(2),y_c(1),z_c(1); x_c(2),y_c(2),z_c(1); x_c(1),y_c(2),z_c(1);
       x_c(1),y_c(1),z_c(2); x_c(2),y_c(1),z_c(2); x_c(2),y_c(2),z_c(2); x_c(1),y_c(2),z_c(2)];
    
    % Faces (all 6)
    f=[1 2 3 4; 2 6 7 3; 6 5 8 7; 5 1 4 8; 1 5 6 2; 4 3 7 8];
    
    patch(ax, 'Vertices', v, 'Faces', f, ...
              'FaceColor', color, 'FaceAlpha', alpha, ...
              'EdgeColor', 'k', 'LineStyle', '-');
end

