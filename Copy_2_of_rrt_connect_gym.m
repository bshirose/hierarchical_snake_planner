% This script computes an optimal trajectory for a 16-link snake robot to
% perform a sequence of "REACH" and "PULL" maneuvers, enabling locomotion.

clear; close all;
rng('default');
use_rrt=true;
n_way=11;
do_col_check=true;
if use_rrt
    do_col_check = true;
end

% =========================================================================
% 1. DEFINE ROBOT AND PROBLEM PARAMETERS
% =========================================================================
MAX_TORQUE = 10.0;       % Maximum allowable torque (Nm) - ADJUST THIS
LINK_MASS_KG = 0.250;   % Mass per link (kg)
RRT_DT = 0.2;           % Virtual time step between RRT nodes (seconds) for derivative calc
n_links = 16; % The robot has 16 joints and 16 links.
link_length = 5.105; % module length (cm)

% --- Locomotion and Waypoint Definitions ---
% 'locomotion_modes' defines the state machine for the gait.
locomotion_modes = {'REACH','REACH','REACH','PULL','PULL','PULL','PULL','PULL','REACH','REACH','REACH'}; 

% 'waypoints' defines the 3D target for the *end-effector* of that segment.
% - For 'REACH': This is the target for the TAIL.
% - For 'PULL': This is the target for the HEAD.
waypoints = { 
        [25,-11,25]',
        [25,-21,25]',
        [25,-31,25]',
        [-5,0,3]',
        [2,0,3]',
        [48,10,45]',
        [48,10,60]',
        [48,10,65]',
        [25,-27,25]',
        [25,-17,25]',
        [25,-7,25]',
    };

% --- Constraint Definitions per Segment ---
% 'num_straight_links' defines end-effector shape constraints per segment.
num_straight_links = {2,5,5,4,3,4,4,4,0,0,0};  
% 'passage_axes' defines the orientation constraint per segment.
passage_axes = {'Y', 'Y','Y','X','X','Z','Z','Z','Y','Y','Y'};

% Define 'positive' or 'negative' direction along the passage_axis.
passage_directions = {'negative', 'negative','negative','negative','negative', 'positive','positive','positive','negative','negative','negative'}; % NEW
% --- Fixed Link Constraint ---
% Define how many links from the base should be fixed (no joint movement).
num_fixed_links_from_base = {4, 4,4, 3,3,3,4,4,4,4,4,4,4};

% --- Initial Problem State ---
q_initial = zeros(1, n_links); 
% q_initial=[pi/10 0 -2*pi/10 0 pi/10 zeros(1,n_links-5)];
q_initial(1:5)=[pi/10 0 -2*pi/10 0 pi/10];
q_initial(7)= -pi/5;
% 'T_base_initial' is the starting pose of the HEAD at t=0.
T_base_initial = eye(4); 
T_base_initial(1, 4) = -20; % Set initial X position to -20
T_base_initial(2, 4) = -2;
T_base_initial(3, 4) = 3;
% --- Environment ---
obstacle_cuboid = [];
safety_margin = 0.01; 
snake_thickness = 2.5;
enable_collision_avoidance = true;
actual_obs=[-20 6-snake_thickness 0 0 6+snake_thickness 10;
            -20 -6-snake_thickness 0 0 -6+snake_thickness 10;
            17-snake_thickness -35 22 17+snake_thickness -10 32;
            31-snake_thickness -35 22 31+snake_thickness -10 32;
            44 4-snake_thickness 45 54 4+snake_thickness 65;
            44 16-snake_thickness 45 54 16+snake_thickness 65;
            19.5 -35 21 30.5 -10 22; % 1st floor
            -30 -30 -5 50 50 0; % floor main
            -20 -5.5 10 0 5.5 11; % zero roof
            19.5 -35 32 30.5 -10 33; % 1st roof
            53 4.5 45 54 15.5 65;
            44 4.5 45 45 15.5 65
            ];
numobs = height(actual_obs);
% --- Useless ---
dummy_cuboids = {
    struct('dims', [22.5, -10, 5, 32.5, -35, 15], 'axis', 'Y'), ...
    struct('dims', [-20, -5, -5, 0, 5, 5], 'axis', 'X'), ...
    struct('dims', [-20, 7, 15, 7, 17, 25], 'axis', 'X')};
              
% just for only IK solution
N = 50;

% State and Control Limits
q_min = -pi/2; q_max = pi/2;
u_min = -5;    u_max = 5;

% --- Visualization Parameters ---
record_video = true;

% rrt params
step_size =0.1;
max_iter = 100000;
max_smoothen=250;






% =========================================================================
% 2. PREVIEW INITIAL SETUP
% =========================================================================
fprintf('Displaying initial problem setup for sanity check...\n');
fig_preview = figure('Name', 'Optimization Setup Preview', 'NumberTitle', 'off');
set(fig_preview, 'WindowState', 'maximized');
ax_preview = axes(fig_preview); grid(ax_preview, 'on'); axis(ax_preview, 'equal'); hold(ax_preview, 'on');
for yy = 1:length(waypoints), plot3(ax_preview, waypoints{yy}(1), waypoints{yy}(2), waypoints{yy}(3), 'r*', 'MarkerSize', 15, 'LineWidth', 2, 'DisplayName', sprintf('Waypoint %d',yy)); end
plot_cuboid_faces(ax_preview, obstacle_cuboid, passage_axes{1});
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
max_range=0; if ~isempty(obstacle_cuboid), max_range=max(abs(obstacle_cuboid)); end; for yy=1:length(waypoints),max_range=max([max_range,abs(waypoints{yy}(:))']);end; max_range=max([max_range,abs(P_initial(:)')]);
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
full_q_opt = q_initial'; 
full_q_dot_opt = zeros(n_links, 1); 
full_u_opt = zeros(n_links, 1); 
full_time_vector = 0;

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

if strcmpi(locomotion_modes{1}, 'PULL')
    fprintf('First move is PULL. Setting initial base to the tail position.\n');
    [~, T_tail_initial] = forwardKinematics(q_initial', link_length, T_base_initial, 'HEAD_BASE');
    T_base_current = T_tail_initial;
end

n_states = 2 * n_links;
n_controls = n_links;
    
curr_path = [];
full_smooth_path=[];
locs=[];
spath=[];
% --- Main Locomotion Loop ---
for i = 1:n_way
    % Get parameters for the current segment
    if i>1 
        if ~strcmp(locomotion_modes{i},locomotion_modes{i-1}) 
            a.curr_path=curr_path;
            a.T_base_current=T_base_current_prev;
            a.base_end=base_end;
            locs=[locs;a];
            curr_path = [];
        else
            
        end
    end
    mode = locomotion_modes{i}; 


    xyz_goal = waypoints{i}; 
    num_links_str = num_straight_links{i};
    current_passage_axis = passage_axes{i}; 
    current_passage_direction = passage_directions{i}; % NEW
    num_fixed = num_fixed_links_from_base{i}; 
    
    % Determine Base and End-Effector for this segment
    if strcmpi(mode, 'REACH'), base_end = 'HEAD_BASE'; else, base_end = 'TAIL_BASE'; end
    fprintf('\n--- Optimizing Segment %d: %s to Waypoint [%s] ---\n', i, mode, num2str(xyz_goal'));
    
    % Store segment info for visualization script
    full_T_base_trajectory{i} = T_base_current; 
    full_base_end_trajectory{i} = base_end;
    
    % --- Optimization Setup ---
    T_guess = 2; 
    limits = [q_min*ones(n_links, 1), q_max*ones(n_links, 1)]; % n_links x 2
    
    
    % --- STAGE 1: Solve Static IK for a valid q_final ---
    q_initial_segment = current_x_initial(1:n_links);

    % if strcmp(mode,"REACH")
    %     q_initial_segment(1:4)=[pi/10 0 -2*pi/10 0];
    % else
    %     q_initial_segment(n_links-5:n_links)=[pi/10 0 -2*pi/10 0 pi/10];
    % end
    [q_final_target, ik_success] = findFinalQ(q_initial_segment, xyz_goal, link_length, T_base_current, ...
                                base_end, 1, current_passage_axis, current_passage_direction, ...
                                num_fixed, limits, actual_obs, numobs, safety_margin, enable_collision_avoidance); 

    [q_final_target, ik_success] = findFinalQ(q_final_target, xyz_goal, link_length, T_base_current, ...
                                base_end, num_links_str, current_passage_axis, current_passage_direction, ...
                                num_fixed, limits, actual_obs, numobs, safety_margin, enable_collision_avoidance); 
    is_free =is_collision_free_RRT(q_final_target, link_length, T_base_current, base_end, actual_obs,numobs);
    is_freew =is_collision_free_RRT(q_initial_segment, link_length, T_base_current, base_end, actual_obs,numobs);
    if ~is_freew
        disp("problem")
    end
    if ~ik_success || ~is_free
        for o=1:100
            limits_min = limits(:, 1);
            limits_range = (limits(:, 2) - limits(:, 1));
            q_rand = limits_min + limits_range .* rand(n_links, 1);
            if  strcmp(mode,"REACH")
                q_rand(1:num_fixed) =q_initial_segment(1:num_fixed);
            else
                q_rand(n_links-num_fixed+1:n_links) =q_initial_segment(n_links-num_fixed+1:n_links);
            end
            [q_final_target, ik_success] = findFinalQ(q_rand, xyz_goal, link_length, T_base_current, ...
                                base_end, num_links_str, current_passage_axis, current_passage_direction, ...
                                num_fixed, limits, actual_obs, numobs, safety_margin, enable_collision_avoidance); 
            is_free =is_collision_free_RRT(q_final_target, link_length, T_base_current, base_end, actual_obs,numobs);
            if ik_success 
                if is_free && do_col_check
                    break
                end
                if ~do_col_check
                    break
                end
            end
            ik_success=false;
        end
    end
    % --- STAGE 2: Create Seed Trajectory (Z0) ---
    if ik_success
        fprintf('IK solver found valid q_final.\n');
        if ~use_rrt
            % 1. Create q_guess by linspace-ing each joint
            q_guess = zeros(n_links, N+1);
            for j = 1:n_links
                q_guess(j, :) = linspace(q_initial_segment(j), q_final_target(j), N+1);
            end
            
            % 2. Estimate q_dot_guess from q_guess (finite difference)
            h_guess = T_guess / N;
            q_dot_guess = [zeros(n_links, 1), diff(q_guess, 1, 2) / h_guess];
            
            % 3. Estimate u_guess (q_ddot) from q_dot_guess (finite difference)
            u_guess = [zeros(n_links, 1), diff(q_dot_guess, 1, 2) / h_guess];
            u_guess(:, end) = u_guess(:, end-1); % Pad last column
            u_guess = max(u_min, min(u_max, u_guess)); % Clamp to limits
            
            % 4. Build x_guess and Z0
            x_guess = [q_guess; q_dot_guess]; % n_states x (N+1)
            Z0 = [x_guess(:); u_guess(:); T_guess];
        end
    else
        % --- Fallback: Use old "stay-in-place" guess ---
        fprintf('WARNING: IK solver failed. Using default "stay-in-place" guess.\n');
        % error("infeasible next pose!")
        break
    end
    if use_rrt
        if strcmp(mode,"REACH")
            path = RRT_Connect_snake_reach(q_initial_segment, q_final_target,num_fixed, link_length, T_base_current, ...
                                     base_end, limits, actual_obs,numobs, ...
                                     max_iter, step_size, MAX_TORQUE, LINK_MASS_KG, RRT_DT);
            if abs(sum(path(1,:)'-q_initial_segment))>0.001
                path=flipud(path);
            end
            if i~=n_way
                lastpoint = path(end,:);
                lastpoint(n_links-2:n_links)=[-pi/5 0 2*pi/5];
                if ~strcmp(locomotion_modes{i+1},mode)
                    path=[path;lastpoint];
                end
            else
                lastpoint = path(end,:);
                lastpoint(n_links-4:n_links)=[pi/10 0 -pi/10 -pi/5 0];
                % path=[path;lastpoint];
            end
            
            curr_path = [curr_path;path];
            % if ~isempty(path)
            %     path = smooth_path_shortcut(path, link_length, T_base_current, base_end, actual_obs, numobs, max_smoothen);
            % end
        elseif strcmp(mode,"PULL")
            path = RRT_Connect_snake_pull(q_initial_segment, q_final_target,num_fixed, link_length, T_base_current, ...
                                     base_end, limits, actual_obs,numobs, ...
                                     max_iter, step_size, MAX_TORQUE, LINK_MASS_KG, RRT_DT);
            if abs(sum(path(1,:)'-q_initial_segment))>0.001
                path=flipud(path);
            end
            if i~=n_way
                lastpoint = path(end,:);
                lastpoint(1:5)=[pi/10 0 pi/10 0 pi/10];
                if ~strcmp(locomotion_modes{i+1},mode)
                    path=[path;lastpoint];
                end
            else
                lastpoint = path(end,:);
                lastpoint(1:5)=[-pi/5 0 2*pi/5];
                path=[path;lastpoint];
            end
            curr_path = [curr_path;path];
            % if ~isempty(path)
            %     path = smooth_path_shortcut(path, link_length, T_base_current, base_end, actual_obs, numobs, max_smoothen);
            % end
        end
        if ~isempty(path)
            M = size(path, 1);
            N=height(path);
            xi = linspace(1, M, N+1);
            q_guess_rrt = interp1(1:M, path, xi, 'linear')'; % Transpose to get n_links x (N+1)
            time_taken = RRT_DT *N;
            % Estimate q_dot from q_guess_rrt (finite difference)
            h_guess = RRT_DT;
            q_dot_guess_rrt = [zeros(n_links, 1), diff(q_guess_rrt, 1, 2) / h_guess]; % Forward diff
            
            % Estimate u (q_ddot) from q_dot_guess_rrt (finite difference)
            u_guess_rrt = [zeros(n_links, 1), diff(q_dot_guess_rrt, 1, 2) / h_guess];
            u_guess_rrt(:, end) = u_guess_rrt(:, end-1); % Pad last column
            
            % Clamp u_guess to be within limits
            u_guess_rrt = max(u_min+0.1, min(u_max-0.1, u_guess_rrt));
            
            % Re-build x_guess and Z0
            x_guess = [q_guess_rrt; q_dot_guess_rrt]; % n_states x (N+1)
            Z0 = [x_guess(:); u_guess_rrt(:); time_taken];
           
        else
            if enable_collision_avoidance
                fprintf('RRT failed. Using default kinematic guess.\n');
                break
            end
        end
    end
    
    
    Z_opt=Z0;

    
    % --- Unpack Results ---
    x_opt = reshape(Z_opt(1:(n_states*(N+1))), n_states, N+1); 
    u_opt = reshape(Z_opt((n_states*(N+1)+1):end-1), n_controls, N+1);
    T_opt = Z_opt(end); 
    q_opt = x_opt(1:n_links, :);
    
    % --- Update State for Next Segment ---
    current_x_initial = x_opt(:, end);
    [~, T_ee_final] = forwardKinematics(q_opt(:, end), link_length, T_base_current, base_end);
    
    % Only update the base if the *next* segment is a different mode.
    if i<length(locomotion_modes) && ~strcmpi(locomotion_modes{i},locomotion_modes{i+1})
        T_base_current_prev=T_base_current;
        T_base_current = T_ee_final; 
    end
    
    % --- Append results to full trajectory ---
    time_segment = linspace(0, T_opt, N+1); 
    cumulative_time_offset = full_time_vector(end);
    full_q_opt=[full_q_opt,q_opt(:,2:end)]; 
    full_q_dot_opt=[full_q_dot_opt,x_opt(n_links+1:end,2:end)];
    full_u_opt=[full_u_opt,u_opt(:,2:end)]; 
    full_time_vector=[full_time_vector,time_segment(2:end)+cumulative_time_offset];
    segment_end_times = [segment_end_times, full_time_vector(end)];
end
a.curr_path=curr_path;
a.T_base_current=T_base_current;
a.base_end=base_end;
locs=[locs;a];

% --- Initialize Full Trajectory Storage ---
smooth_full_q_opt = q_initial'; 
smooth_full_time_vector = 0;

% --- Store poses for visualization ---
smooth_full_T_base_trajectory = {};
smooth_full_base_end_trajectory = {};
smooth_segment_end_times = [0]; 
t=1;
for i=1:height(locs)
    
    [spath,skipped_counts] = smooth_path_shortcut(locs(i).curr_path, link_length, locs(i).T_base_current, locs(i).base_end, actual_obs, numobs, max_smoothen);
    disp("path length reduction")
    size(locs(i).curr_path)
    size(spath)
    for j=1:height(spath)
        if j==1
            t=t+RRT_DT;
        else
            t=t+(skipped_counts(j-1)+1)*RRT_DT;
        end
        smooth_full_q_opt=[smooth_full_q_opt ,spath(j,:)'];
        smooth_full_time_vector=[smooth_full_time_vector ,t];
    end
    smooth_full_T_base_trajectory{i}=locs(i).T_base_current;
    smooth_full_base_end_trajectory{i}=locs(i).base_end;
    smooth_segment_end_times=[smooth_segment_end_times,t];
    
end

% =========================================================================
% 4. SAVE RESULTS TO FILE
% =========================================================================
    output_filename = 'trajectory_data_v15_gym.mat';
    % Save all new data required for v14 visualization
    save(output_filename, 'full_q_opt', 'full_q_dot_opt', 'full_u_opt', 'full_time_vector', ...
        'link_length', 'n_links', 'waypoints', 'record_video', ...
        'obstacle_cuboid', 'passage_axes', 'num_straight_links', 'num_fixed_links_from_base', ...
        'enable_collision_avoidance', 'dummy_cuboids', 'T_base_initial', ...
        'locomotion_modes', 'full_T_base_trajectory', 'full_base_end_trajectory', 'segment_end_times', ...
        'passage_directions','actual_obs','numobs','smooth_full_q_opt','smooth_full_time_vector','smooth_full_T_base_trajectory', ...
        'smooth_full_base_end_trajectory','smooth_segment_end_times'); % NEW: Save directions
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
% 
% 
% function smoothed_path = smooth_path_shortcut(path, link_length, T_base, base_end, obstacles, numobs, max_iterations)
%     % Initialize
%     smoothed_path = path;
% 
%     if size(smoothed_path, 1) < 3
%         return;
%     end
% 
% 
%     for k = 1:max_iterations
%         % 1. Pick two random indices (i and j)
%         len = size(smoothed_path, 1);
%         if len < 4, break; end
% 
%         idx1 = randi(len - 3);
%         idx2 = randi([idx1 + 2, len-1]); % Ensure idx2 is at least 2 steps ahead
% 
%         q1 = smoothed_path(idx1, :);
%         q2 = smoothed_path(idx2, :);
% 
%         % 2. Check if straight line connection is valid
%         % We check discrete points along the line segment
%         valid_shortcut = true;
%         steps = 100; % Check 10 points along the shortcut
%         for t = linspace(0, 1, steps)
%             q_check = (1-t)*q1 + t*q2;
%             if ~is_collision_free_RRT(q_check', link_length, T_base, base_end, obstacles, numobs)
%                 valid_shortcut = false;
%                 break;
%             end
%         end
% 
%         % 3. If valid, remove intermediate nodes
%         if valid_shortcut
%             % Replace path with [start ... q1, q2 ... end]
%             smoothed_path = [smoothed_path(1:idx1, :); smoothed_path(idx2:end, :)];
%         end
%     end
% end
% 

function [q_final, success] = findFinalQ(q_guess, xyz_goal, link_length, T_base, base_end, num_straight_links, passage_axis, passage_direction, num_fixed_links, limits, actual_obs, numobs, safety_margin, enable_collision_avoidance)
    
    n_links = length(q_guess);
    
    % Cost function: Minimize joint-space distance from the initial guess
    cost_ik = @(q) sum((q - q_guess).^2);
    

    constraints_ik = @(q) staticConstraints(q, n_links, q_guess, xyz_goal, link_length, ...
                                            actual_obs, numobs, safety_margin, passage_axis, ... % MODIFIED
                                            passage_direction, num_straight_links, ...
                                            enable_collision_avoidance, T_base, base_end, num_fixed_links);
    % Bounds
    lb = limits(:, 1);
    ub = limits(:, 2);
    
    % Options for the IK solver (less strict than main optimization)
    options_ik = optimoptions('fmincon', 'Algorithm', 'interior-point','ScaleProblem', true, ...
        'Display', 'none', 'OptimalityTolerance', 1e-3, ...
        'ConstraintTolerance', 1e-4, 'EnableFeasibilityMode', true);
    
    [q_final, ~, exitflag] = fmincon(cost_ik, q_guess, [], [], [], [], lb, ub, constraints_ik, options_ik);
    
    % Check if solver was successful
    success = (exitflag > 0);
end

function [c_ik, ceq_ik] = staticConstraints(q, n_links, q_initial, xyz_goal, link_length, actual_obs, numobs, safety_margin, passage_axis, passage_direction, num_straight_links, enable_collision_avoidance, T_base, base_end, num_fixed_links) % MODIFIED
    % --- 1. Final Position Constraint ---
    [P_final, T_ee_final] = forwardKinematics(q, link_length, T_base, base_end);
    ceq_pos_end = T_ee_final(1:3, 4) - xyz_goal;
    
    % --- 2. Shape Constraints (MODIFIED for Direction) ---
    c_orientation_direction = []; % NEW inequality constraint
    ceq_collinearity = []; 
    ceq_orientation = [];
    if base_end == "HEAD_BASE"
        if num_straight_links > 0
            % ceq_collinearity=q(n_links-num_straight_links+1:n_links);
            num_points = size(P_final, 2);
            final_link_vec = P_final(:, num_points) - P_final(:, num_points-1);
            if num_straight_links > 1
                for k=(num_points-num_straight_links+1):(num_points-1)
                    ceq_collinearity=[ceq_collinearity; cross(final_link_vec, P_final(:,k+1)-P_final(:,k))]; 
                end
            end
            % Constrain components based on the passage_axis and direction
          switch upper(passage_axis)
                case 'X'
                    ceq_orientation=final_link_vec(2:3); % Y and Z must be 0
                    if strcmpi(passage_direction, 'positive')
                        c_orientation_direction = -final_link_vec(1); % Enforce -X <= 0  (so X >= 0)
                    else % 'negative'
                        c_orientation_direction = final_link_vec(1);  % Enforce X <= 0
                    end
                case 'Y'
                    ceq_orientation=final_link_vec([1,3]); % X and Z must be 0
                    if strcmpi(passage_direction, 'positive')
                        c_orientation_direction = -final_link_vec(2); % Enforce -Y <= 0  (so Y >= 0)
                    else % 'negative'
                        c_orientation_direction = final_link_vec(2);  % Enforce Y <= 0
                    end
                case 'Z'
                    ceq_orientation=final_link_vec(1:2); % X and Y must be 0
                    if strcmpi(passage_direction, 'positive')
                        c_orientation_direction = -final_link_vec(3); % Enforce -Z <= 0  (so Z >= 0)
                    else % 'negative'
                        c_orientation_direction = final_link_vec(3);  % Enforce Z <= 0
                    end
            end
        end

    elseif strcmpi(base_end, 'TAIL_BASE') % --- CORRECTED LOGIC ---
        if num_straight_links > 0
            num_points = size(P_final, 2);
            ceq_collinearity=q(1:num_straight_links);
            % --- FIX 1 ---
            % The end-effector is the HEAD.
            % We must get the vector of the *first* link (the head).
            final_link_vec = P_final(:, 2) - P_final(:, 1);
            
            switch upper(passage_axis)
                case 'X'
                    ceq_orientation=final_link_vec(2:3); % Y and Z must be 0
                    if strcmpi(passage_direction, 'negative')
                        c_orientation_direction = -final_link_vec(1); % Enforce -X <= 0  (so X >= 0)
                    else % 'negative'
                        c_orientation_direction = final_link_vec(1);  % Enforce X <= 0
                    end
                case 'Y'
                    ceq_orientation=final_link_vec([1,3]); % X and Z must be 0
                    if strcmpi(passage_direction, 'negative')
                        c_orientation_direction = -final_link_vec(2); % Enforce -Y <= 0  (so Y >= 0)
                    else % 'negative'
                        c_orientation_direction = final_link_vec(2);  % Enforce Y <= 0
                    end
                case 'Z'
                    ceq_orientation=final_link_vec(1:2); % X and Y must be 0
                    if strcmpi(passage_direction, 'negative')
                        c_orientation_direction = -final_link_vec(3); % Enforce -Z <= 0  (so Z >= 0)
                    else % 'negative'
                        c_orientation_direction = final_link_vec(3);  % Enforce Z <= 0
                    end
            end
        end
    end 
    % --- 3. Fixed Links (Static Position) Constraint ---
    ceq_fixed_joints_static = [];
    if num_fixed_links > 0
        if strcmpi(base_end, 'HEAD_BASE')
            fixed_indices = 1:num_fixed_links;
        else % 'TAIL_BASE'
            fixed_indices = (n_links - num_fixed_links + 1):n_links;
        end
        q_fixed_current = q(fixed_indices);
        q_fixed_initial = q_initial(fixed_indices);
        ceq_fixed_joints_static = q_fixed_current - q_fixed_initial;
    end
    
    % --- 4. Collision Avoidance ---
    c_obs=[];
    % if enable_collision_avoidance
    %     for j=1:n_links+1 % Check every link point
    %         c_obs=[c_obs; get_cuboid_violations(P_final(:,j), obstacle_cuboid, safety_margin, passage_axis)];
    %     end
    % end
    if enable_collision_avoidance && ~isempty(actual_obs)
        n_links_local = size(P_final, 2) - 1; % Should be same as n_links
        
        for o = 1:numobs
            obs = actual_obs(o, :);
            % Get box center and half-dimensions (d)
            center = [(obs(1)+obs(4))/2; (obs(2)+obs(5))/2; (obs(3)+obs(6))/2];
            dims_half = [obs(4)-center(1); obs(5)-center(2); obs(6)-center(3)];
            
            % Add safety margin to the box dimensions
            dims_half_safe = dims_half + safety_margin;
            
            % Check each link segment
            for i = 1:n_links_local
                p1 = P_final(:, i);
                p2 = P_final(:, i+1);
                
                % Check 5 interpolated points (t=0, 0.25, 0.5, 0.75, 1.0)
                % This mimics the RRT checker's interpolation
                for t = 0:0.25:1 
                    p = (1-t)*p1 + t*p2; % Interpolated point
                    
                    % Calculate Signed Distance Function (SDF)
                    q_vec = abs(p - center) - dims_half_safe;
                    sdf = norm(max(q_vec, 0)) + min(max(q_vec(1), max(q_vec(2), q_vec(3))), 0);
                    
                    % We want sdf >= 0 (safe).
                    % fmincon wants c_ik <= 0 (safe).
                    % So, we set c_ik = -sdf.
                    % c_obs = [c_obs; -sdf];
                end
            end
        end
    end
    
    % --- Assemble Final Constraint Vector ---
    c_ik = [c_obs(:); c_orientation_direction(:)];
    
    ceq_ik = [ceq_pos_end(:); ceq_collinearity(:); ceq_orientation(:); ceq_fixed_joints_static(:)];
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
function violations = get_cuboid_violations(p, cuboid, margin, axis)
    if isempty(cuboid)
        violations = []; 
        return;
    end
    x_min=cuboid(1);y_min=cuboid(2);z_min=cuboid(3);x_max=cuboid(4);y_max=cuboid(5);z_max=cuboid(6); violations=[-1;-1;-1;-1]; 
    switch upper(axis), case 'X',if p(1)>x_min&&p(1)<x_max,violations=[(y_min+margin)-p(2);p(2)-(y_max-margin);(z_min+margin)-p(3);p(3)-(z_max-margin)];end; case 'Y',if p(2)>y_min&&p(2)<y_max,violations=[(x_min+margin)-p(1);p(1)-(x_max-margin);(z_min+margin)-p(3);p(3)-(z_max-margin)];end; case 'Z',if p(3)>z_min&&p(3)<z_max,violations=[(x_min+margin)-p(1);p(1)-(x_max-margin);(y_min+margin)-p(2);p(2)-(y_max-margin)];end; end
end
% --- Helper: Plot Cuboid (Identical to v10.3) ---
function plot_cuboid_faces(ax,c,p), if isempty(c),return;end; x_c=[c(1),c(4)];y_c=[c(2),c(5)];z_c=[c(3),c(6)]; v=[x_c(1),y_c(1),z_c(1);x_c(2),y_c(1),z_c(1);x_c(2),y_c(2),z_c(1);x_c(1),y_c(2),z_c(1);x_c(1),y_c(1),z_c(2);x_c(2),y_c(1),z_c(2);x_c(2),y_c(2),z_c(2);x_c(1),y_c(2),z_c(2)]; f_Xm=[1 4 8 5];f_XM=[2 3 7 6];f_Ym=[1 2 6 5];f_YM=[3 4 8 7];f_Zm=[1 2 3 4];f_ZM=[5 6 7 8];
switch upper(p), case 'X',fs=[f_Ym;f_YM;f_Zm;f_ZM];case 'Y',fs=[f_Xm;f_XM;f_Zm;f_ZM];case 'Z',fs=[f_Xm;f_XM;f_Ym;f_YM];otherwise,fs=[f_Ym;f_YM;f_Zm;f_ZM];end; patch(ax,'Vertices',v,'Faces',fs,'FaceColor','b','FaceAlpha',0.1,'EdgeColor','b','LineStyle','--'); end
% --- Helper: Homogeneous Transforms (Identical to v10.3) ---
function T=Rz(t),c=cos(t);s=sin(t);T=[c -s 0 0;s c 0 0;0 0 1 0;0 0 0 1];end 
function T=Ry(t),c=cos(t);s=sin(t);T=[c 0 s 0;0 1 0 0;-s 0 c 0;0 0 0 1];end
function T=Tx(d),T=[1 0 0 d;0 1 0 0;0 0 1 0;0 0 0 1];end

% =========================================================================
% NEW (v16): RRT-Connect Implementation and Helpers
% =========================================================================

function path = RRT_Connect_snake_reach(q_start, q_goal,num_fixed, link_length, T_base, ...
                                 base_end, limits, obstacle_cuboid,numobs, ...
                                 max_iter, step_size, max_torque, link_mass, dt)

    reach=1;
    % fprintf('RRT-Connect: Searching for path from q_start to q_goal...\n');
    n_links = length(q_start);
    path = []; % Initialize empty path

    % Initialize two trees
    tree_a.nodes = q_start(:)';      % From start
    tree_a.parents = 0;
    tree_b.nodes = q_goal(:)';       % From goal
    tree_b.parents = 0;

    % Pre-calculate for sampling
    limits_min = limits(:, 1);
    limits_range = (limits(:, 2) - limits(:, 1));

    first = true;

    for i = 1:max_iter
        % 1. Sample a random configuration
        q_rand = limits_min + limits_range .* rand(n_links, 1);
        if first
            q_rand = tree_a.nodes(1,:)';
            first = false;
        end
        q_rand(1:num_fixed) =tree_a.nodes(1,1:num_fixed);
        % 2. Extend tree_a towards q_rand
        dists_a = vecnorm(tree_a.nodes - q_rand', 2, 2);
        [~, nearest_idx_a] = min(dists_a);
        q_near_a = tree_a.nodes(nearest_idx_a, :)';
        
        [q_new, status_a] = steer(q_near_a, q_rand, step_size,link_length, T_base, base_end, obstacle_cuboid);
        
        % Check for collision
        if ~is_collision_free_RRT(q_new, link_length, T_base, base_end, obstacle_cuboid,numobs)
            % 'Trapped'
            status_a = 'Trapped';
        end
        if ~strcmp(status_a, 'Trapped')
             if check_torque_violation(q_new, nearest_idx_a, tree_a, link_length, link_mass, dt, max_torque,num_fixed,reach)
                 status_a = 'Trapped'; % Torque limit exceeded
             end
        end

        % If we advanced, add the new node to tree_a
        if ~strcmp(status_a, 'Trapped')
            tree_a.nodes = [tree_a.nodes; q_new'];
            tree_a.parents = [tree_a.parents; nearest_idx_a];
            q_new_idx_a = size(tree_a.nodes, 1);
            q_target = q_new; % This is the new target for tree_b

            % 3. Connect tree_b towards q_new
            dists_b = vecnorm(tree_b.nodes - q_target', 2, 2);
            [~, nearest_idx_b] = min(dists_b);
            q_near_b = tree_b.nodes(nearest_idx_b, :)';
            
            [q_connect, status_b] = steer(q_near_b, q_target, step_size,link_length, T_base, base_end, obstacle_cuboid);
            % --- DYNAMIC CHECK B (Initial) ---
            if ~strcmp(status_b, 'Trapped')
                if check_torque_violation(q_connect, nearest_idx_b, tree_b, link_length, link_mass, dt, max_torque,num_fixed,reach)
                    status_b = 'Trapped';
                end
            end
            if ~is_collision_free_RRT(q_connect, link_length, T_base, base_end, obstacle_cuboid,numobs)
                status_b = 'Trapped';
            end
            
            % If valid, add to tree_b and keep trying to connect
            if ~strcmp(status_b, 'Trapped')
                tree_b.nodes = [tree_b.nodes; q_connect'];
                tree_b.parents = [tree_b.parents; nearest_idx_b];
                q_connect_idx_b = size(tree_b.nodes, 1);
                
                q_current_b = q_connect;
                q_current_b_idx = q_connect_idx_b;
                
                % 4. Iterative Connection
                while strcmp(status_b, 'Advanced') % Keep trying until 'Reached' or 'Trapped'
                    [q_next_b, status_b] = steer(q_current_b, q_target, step_size,link_length, T_base, base_end, obstacle_cuboid);
                    if ~strcmp(status_b, 'Trapped')
                        if check_torque_violation(q_next_b, q_current_b_idx, tree_b, link_length, link_mass, dt, max_torque,num_fixed,reach)
                            status_b = 'Trapped';
                        end
                    end
                    if ~is_collision_free_RRT(q_next_b, link_length, T_base, base_end, obstacle_cuboid,numobs)
                        status_b = 'Trapped'; % Failed to connect
                        break;
                    end
                    
                    % Add the valid intermediate node
                    tree_b.nodes = [tree_b.nodes; q_next_b'];
                    tree_b.parents = [tree_b.parents; q_current_b_idx];
                    
                    q_current_b = q_next_b;
                    q_current_b_idx = size(tree_b.nodes, 1);
                end % while status_b == 'Advanced'
                
                % 5. Check for Success
                if strcmp(status_b, 'Reached')
                    fprintf('RRT-Connect: Trees connected at iteration %d!\n', i);
                    path_a = construct_path(tree_a, q_new_idx_a);
                    path_b = construct_path(tree_b, q_current_b_idx);
                    
                    % Combine paths (path_b is from goal, so flip it)
                    path = [path_a; flipud(path_b)]; 
                    fprintf('RRT-Connect: Path found with %d waypoints.\n', size(path, 1));
                    return; % Success!
                end
            end
        end
        
        % 6. Swap Trees
        temp = tree_a;
        tree_a = tree_b;
        tree_b = temp;
        
        if mod(i, 5000) == 0
            fprintf('RRT-Connect Iter: %d, Tree A: %d nodes, Tree B: %d nodes\n', i, size(tree_a.nodes, 1), size(tree_b.nodes, 1));
        end
    end % for max_iter

    fprintf('RRT-Connect: Failed to find a path after %d iterations.\n', max_iter);
end

function path = RRT_Connect_snake_pull(q_start, q_goal,num_fixed, link_length, T_base, ...
                                 base_end, limits, obstacle_cuboid,numobs, ...
                                 max_iter, step_size, max_torque, link_mass, dt)
% RRT_Connect_snake: Finds a collision-free path in C-space.
%
%   - q_start, q_goal: (n_links x 1) joint vectors
%   - limits: (n_links x 2) matrix of [q_min, q_max]
%   - obstacle_cuboid: The 'acutal_obs' [xmin, ymin, zmin, xmax, ymax, zmax]

    % fprintf('RRT-Connect: Searching for path from q_start to q_goal...\n');
    reach=0;
    n_links = length(q_start);
    path = []; % Initialize empty path

    % Initialize two trees
    tree_a.nodes = q_start(:)';      % From start
    tree_a.parents = 0;
    tree_b.nodes = q_goal(:)';       % From goal
    tree_b.parents = 0;

    % Pre-calculate for sampling
    limits_min = limits(:, 1);
    limits_range = (limits(:, 2) - limits(:, 1));
    first = true;
    for i = 1:max_iter
        % 1. Sample a random configuration
        q_rand = limits_min + limits_range .* rand(n_links, 1);
        if first
            q_rand = tree_a.nodes(1,:)';
            first = false;
        end
        q_rand(n_links-num_fixed+1:n_links) =tree_a.nodes(1,n_links-num_fixed+1:n_links);
        % 2. Extend tree_a towards q_rand
        dists_a = vecnorm(tree_a.nodes - q_rand', 2, 2);
        [~, nearest_idx_a] = min(dists_a);
        q_near_a = tree_a.nodes(nearest_idx_a, :)';
        
        [q_new, status_a] = steer(q_near_a, q_rand, step_size,link_length, T_base, base_end, obstacle_cuboid);
        
        % Check for collision
        if ~is_collision_free_RRT(q_new, link_length, T_base, base_end, obstacle_cuboid,numobs)
            % 'Trapped'
            status_a = 'Trapped';
        end
        % --- DYNAMIC CHECK A ---
        if ~strcmp(status_a, 'Trapped')
             if check_torque_violation(q_new, nearest_idx_a, tree_a, link_length, link_mass, dt, max_torque,num_fixed,reach)
                 status_a = 'Trapped'; 
             end
        end

        % If we advanced, add the new node to tree_a
        if ~strcmp(status_a, 'Trapped')
            tree_a.nodes = [tree_a.nodes; q_new'];
            tree_a.parents = [tree_a.parents; nearest_idx_a];
            q_new_idx_a = size(tree_a.nodes, 1);
            q_target = q_new; % This is the new target for tree_b

            % 3. Connect tree_b towards q_new
            dists_b = vecnorm(tree_b.nodes - q_target', 2, 2);
            [~, nearest_idx_b] = min(dists_b);
            q_near_b = tree_b.nodes(nearest_idx_b, :)';
            
            [q_connect, status_b] = steer(q_near_b, q_target, step_size,link_length, T_base, base_end, obstacle_cuboid);
            if ~strcmp(status_b, 'Trapped')
                if check_torque_violation(q_connect, nearest_idx_b, tree_b, link_length, link_mass, dt, max_torque,num_fixed,reach)
                    status_b = 'Trapped'; 
                end
             end
            if ~is_collision_free_RRT(q_connect, link_length, T_base, base_end, obstacle_cuboid,numobs)
                status_b = 'Trapped';
            end
            
            % If valid, add to tree_b and keep trying to connect
            if ~strcmp(status_b, 'Trapped')
                tree_b.nodes = [tree_b.nodes; q_connect'];
                tree_b.parents = [tree_b.parents; nearest_idx_b];
                q_connect_idx_b = size(tree_b.nodes, 1);
                
                q_current_b = q_connect;
                q_current_b_idx = q_connect_idx_b;

                % 4. Iterative Connection
                while strcmp(status_b, 'Advanced') % Keep trying until 'Reached' or 'Trapped'
                    [q_next_b, status_b] = steer(q_current_b, q_target, step_size,link_length, T_base, base_end, obstacle_cuboid);
                    if ~strcmp(status_b, 'Trapped')
                        if check_torque_violation(q_next_b, q_current_b_idx, tree_b, link_length, link_mass, dt, max_torque,num_fixed,reach)
                            status_b = 'Trapped';
                        end
                    end
                    if ~is_collision_free_RRT(q_next_b, link_length, T_base, base_end, obstacle_cuboid,numobs)
                        status_b = 'Trapped'; % Failed to connect
                        break;
                    end
                    
                    % Add the valid intermediate node
                    tree_b.nodes = [tree_b.nodes; q_next_b'];
                    tree_b.parents = [tree_b.parents; q_current_b_idx];
                    
                    q_current_b = q_next_b;
                    q_current_b_idx = size(tree_b.nodes, 1);
                end % while status_b == 'Advanced'
                
                % 5. Check for Success
                if strcmp(status_b, 'Reached')
                    fprintf('RRT-Connect: Trees connected at iteration %d!\n', i);
                    path_a = construct_path(tree_a, q_new_idx_a);
                    path_b = construct_path(tree_b, q_current_b_idx);
                    
                    % Combine paths (path_b is from goal, so flip it)
                    path = [path_a; flipud(path_b)]; 
                    fprintf('RRT-Connect: Path found with %d waypoints.\n', size(path, 1));
                    return; % Success!
                end
            end
        end
        
        % 6. Swap Trees
        temp = tree_a;
        tree_a = tree_b;
        tree_b = temp;
        
        if mod(i, 5000) == 0
            fprintf('RRT-Connect Iter: %d, Tree A: %d nodes, Tree B: %d nodes\n', i, size(tree_a.nodes, 1), size(tree_b.nodes, 1));
        end
    end % for max_iter

    fprintf('RRT-Connect: Failed to find a path after %d iterations.\n', max_iter);
end

% --- RRT Helper 1: Steer Function (With Path Collision Checking) ---
function [q_new, status] = steer(q_near, q_target, step_size, ...
                                 link_length, T_base, base_end, all_obs)
% Steers from q_near towards q_target by step_size AND
% checks the entire new path segment for collisions.
% 'all_obs' is the obstacle variable (e.g., acutal_obs).

    dir = q_target - q_near;
    dist = norm(dir);
    
    % Determine the actual new configuration
    if dist < step_size
        q_new = q_target;
        status = 'Reached';
    else
        q_new = q_near + (dir / dist) * step_size;
        status = 'Advanced';
    end
    
    % --- NEW: Interpolation and Collision Check ---
    % Check N points along the new segment. 
    % We can skip t=0 (q_near) as it's already known to be valid.
    num_interp_checks = 10; % Or ceil(norm(q_new - q_near) / 0.1) for dynamic checks
    t_steps = linspace(0, 1, num_interp_checks);
    
    for i = 2:num_interp_checks
        t = t_steps(i);
        q_interp = (1-t)*q_near + t*q_new;
        
        if ~is_collision_free_RRT(q_interp, link_length, T_base, base_end, all_obs,height(all_obs))
            % Collision found!
            q_new = q_near; % Return the original node (no advance)
            status = 'Trapped'; % Signal failure
            return;
        end
    end
    
    % If loop finishes, the whole path was clear.
    % q_new and status are already set correctly by this point.
end

% --- RRT Helper 2: Path Reconstruction ---
function path = construct_path(tree, leaf_idx)
% Traces from a leaf node back to the root
    path = [];
    current_idx = leaf_idx;
    
    while current_idx > 0
        path = [tree.nodes(current_idx, :); path];
        current_idx = tree.parents(current_idx);
    end
end

% --- RRT Helper 3: Collision Checker ---
function free = is_collision_free_RRT(q, link_length, T_base, base_end, obstacles,numobs)
% Checks if a configuration 'q' collides with the 'obs' cuboid.
% 'obs' is [xmin, ymin, zmin, xmax, ymax, zmax]
    
    free = true; % Assume free
    if isempty(obstacles)
        return;
    end
    
    % Get all joint positions in the world frame
    [P_world, ~] = forwardKinematics(q, link_length, T_base, base_end);
    for o=1:numobs
        obs = obstacles(o,:);
        % Define obstacle boundaries
        x_min=obs(1); y_min=obs(2); z_min=obs(3);
        x_max=obs(4); y_max=obs(5); z_max=obs(6);
        
        n_links = size(P_world, 2) - 1;
        
        % Check each link segment for collision
        for i = 1:n_links
            p1 = P_world(:, i);
            p2 = P_world(:, i+1);
            
            % Check 5 interpolated points along the link (incl. endpoints)
            for t = 0:0.05:1
                p = (1-t)*p1 + t*p2; % Interpolated point
                
                % Check if the point 'p' is *inside* the cuboid
                if (p(1) >= x_min && p(1) <= x_max && ...
                    p(2) >= y_min && p(2) <= y_max && ...
                    p(3) >= z_min && p(3) <= z_max)
                    
                    % Collision detected
                    free = false;
                    return;
                end
            end
        end
    end
end



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

function violates = check_torque_violation(q_new, parent_idx, tree, link_length_cm, mass_kg, dt, max_torque,num_fixed,reach)
    % Checks if the transition to q_new violates torque limits.
    % We need Parent (Current) and Grandparent (Previous) to calc derivatives.
    
    violates = false;
    n_links = length(q_new);
    g_vec = [0;0;-9.81];
    link_length_m = link_length_cm / 100; % Convert to meters for physics
    
    % Get History
    q_curr = tree.nodes(parent_idx, :)'; % The 'parent' in the tree is the current state
    
    parent_of_parent_idx = tree.parents(parent_idx);
    
    if parent_of_parent_idx == 0
        % No grandparent (Start of tree). Assume zero velocity/accel approach.
        q_prev = q_curr; 
    else
        q_prev = tree.nodes(parent_of_parent_idx, :)';
    end
    
    % Finite Difference Derivatives
    % Velocity at current step (approaching q_new)
    q_dot_new = (q_new - q_curr) / dt;
    
    % Velocity at previous step
    q_dot_curr = (q_curr - q_prev) / dt;
    
    % Acceleration required to reach q_dot_new from q_dot_curr
    q_ddot_new = (q_dot_new - q_dot_curr) / dt;
    
    % Calculate Inverse Dynamics (RNEA)
    % We check torques required at the NEW configuration to sustain this motion
    tau = inverseDynamicsRNEA(q_new, q_dot_new, q_ddot_new, n_links, link_length_m, mass_kg, g_vec);

    if reach==1
        tau(1:num_fixed)=zeros(num_fixed,1);
    else
        tau(n_links-num_fixed+1:n_links)=zeros(num_fixed,1);
    end
    
    if any(abs(tau) > max_torque)
        violates = true;
        % Optional debug:
        % fprintf('Torque Violation! Max: %.2f Nm\n', max(abs(tau)));
    end
end

function tau = inverseDynamicsRNEA(q, q_dot, q_ddot, n_links, L, m, g)
    % Recursive Newton-Euler Algorithm
    % L must be in meters, m in kg.
    
    CoM = [L/2; 0; 0]; 
    I_CoM = diag([0, (1/12)*m*L^2, (1/12)*m*L^2]);
    z_axis = [0; 0; 1];
    y_axis = [0; 1; 0];
    
    w = zeros(3, n_links + 1);      
    w_dot = zeros(3, n_links + 1);  
    a = zeros(3, n_links + 1);      
    ac = zeros(3, n_links + 1);     
    
    a(:,1) = -g; 
    
    for i = 1:n_links
        if mod(i, 2) == 1 
            R_world_to_body = [cos(q(i)) -sin(q(i)) 0; sin(q(i)) cos(q(i)) 0; 0 0 1]; % Rz manual
            joint_axis = z_axis;
        else 
            R_world_to_body = [cos(q(i)) 0 sin(q(i)); 0 1 0; -sin(q(i)) 0 cos(q(i))]; % Ry manual
            joint_axis = y_axis;
        end
        R_body_to_world = R_world_to_body'; 
        
        w(:, i+1) = R_body_to_world*w(:,i) + q_dot(i)*joint_axis;
        w_dot(:, i+1) = R_body_to_world*w_dot(:,i) + cross(R_body_to_world*w(:,i), q_dot(i)*joint_axis) + q_ddot(i)*joint_axis;
        
        p_link = [L; 0; 0]; 
        a(:, i+1) = R_body_to_world * (a(:,i) + cross(w_dot(:,i), p_link) + cross(w(:,i), cross(w(:,i), p_link)));
        ac(:, i+1) = a(:,i+1) + cross(w_dot(:,i+1), CoM) + cross(w(:,i+1), cross(w(:,i+1), CoM));
    end
    
    f = zeros(3, n_links + 1); 
    n = zeros(3, n_links + 1); 
    tau = zeros(n_links, 1);   
    
    for i = n_links:-1:1
        F = m * ac(:, i+1);
        N = I_CoM * w_dot(:, i+1) + cross(w(:, i+1), I_CoM * w(:, i+1));
        p_link = [L; 0; 0];
        
        if i == n_links
            f(:,i) = F;
            n(:,i) = N + cross(CoM, F);
        else
            if mod(i+1, 2) == 1
                R_next = [cos(q(i+1)) -sin(q(i+1)) 0; sin(q(i+1)) cos(q(i+1)) 0; 0 0 1]'; 
            else
                R_next = [cos(q(i+1)) 0 sin(q(i+1)); 0 1 0; -sin(q(i+1)) 0 cos(q(i+1))]';
            end
            f(:,i) = R_next * f(:,i+1) + F;
            n(:,i) = N + R_next * n(:,i+1) + cross(CoM, F) + cross(p_link, R_next*f(:,i+1));
        end
        
        if mod(i, 2) == 1, joint_axis = z_axis; else, joint_axis = y_axis; end
        tau(i) = n(:,i)' * joint_axis;
    end
end