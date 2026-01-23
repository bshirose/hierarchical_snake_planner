
clear; close all;

% =========================================================================
% 1. LOAD DATA AND CHECK FOR OPTIONAL VARIABLES
% =========================================================================
global save_execl;
save_execl=false;
% --- Load Data ---
data_filename = 'snake_traj.mat'; % Load the v14 data file
if ~exist(data_filename, 'file')
    error('Could not find %s. Please run the generation script first.', data_filename);
end
fprintf('Loading trajectory data from %s...\n', data_filename);
load(data_filename);


% =========================================================================
% 2. PLOT STATE AND CONTROL TRAJECTORIES (2D GRAPHS)
% =========================================================================
figure('Name', 'Optimal Trajectories (Full Path)', 'NumberTitle', 'off');
q_legend = arrayfun(@(d) sprintf('q%d', d), 1:n_links, 'UniformOutput', false);

animateRobot(full_q_opt, link_length, full_time_vector, waypoints, record_video, obstacle_cuboid, segment_end_times, full_T_base_trajectory, full_base_end_trajectory,actual_obs);

% =========================================================================
% LOCAL FUNCTIONS
% =========================================================================

% --- Main Animation Function (v14) ---
% This is the generalized function that can handle dynamic base poses.
function animateRobot(q_trajectory, link_length, time_vector, waypoints, record_video, obstacle_cuboid, segment_end_times, T_base_traj, base_end_traj,actual_obs)
    global save_execl;    
    % --- Setup the Figure ---
    fig = figure('Name', 'Snake Robot Animation', 'NumberTitle', 'off');
    set(fig, 'WindowState', 'maximized');
    ax = axes(fig); grid(ax, 'on'); axis(ax, 'equal'); hold(ax, 'on');
    pause(2)
    % --- Plot Static Environment ---
    for i = 1:length(waypoints), plot3(ax, waypoints{i}(1), waypoints{i}(2), waypoints{i}(3), 'g*', 'MarkerSize', 15, 'LineWidth', 2, 'DisplayName', sprintf('Waypoint %d',i)); end
    % for k = 1:length(dummy_cuboids), plot_cuboid_faces(ax, dummy_cuboids{k}.dims, dummy_cuboids{k}.axis, 'Dummy Cuboid'); end
    if exist('actual_obs', 'var') && ~isempty(actual_obs)
        num_obs = height(actual_obs);
        for k = 1:num_obs
            % Get the coordinates for the k-th obstacle
            obs_dims = actual_obs(k,:);
            % obs_dims = acutal_obs( (k-1)*6 + 1 : k*6 );
            
            % Plot it using the new function
            plot_solid_cuboid(ax, obs_dims, 'r', 0.5); % Red, 20% opaque
        end
    end
    % --- Plot Initial Robot Pose ---
    % Get the pose and mode for the *first* segment
    [P_initial, ~] = forwardKinematics(q_trajectory(:, 1), link_length, T_base_traj{1}, base_end_traj{1});
    robot_plot = plot3(ax, P_initial(1,:), P_initial(2,:), P_initial(3,:), 'o-', 'LineWidth', 2, 'MarkerSize', 4, 'Color', [0.1, 0.4, 0.8], 'DisplayName', 'Snake Robot');
    
    % --- Set Plot Limits (Using helper) ---
    % 'getAllPoints' calculates the 3D position of all joints at all
    % time steps to find the total bounding box of the animation.
    max_range = 0; if ~isempty(obstacle_cuboid), max_range=max(abs(obstacle_cuboid(:))); end; for i=1:length(waypoints), max_range=max([max_range, abs(waypoints{i}(:))']); end; P_all=getAllPoints(q_trajectory,link_length,T_base_traj,base_end_traj,segment_end_times); max_range=max([max_range, abs(P_all(:)')]);
    lim = ceil(max_range * 1.1); if lim==0, lim=5; end; axis(ax, [-lim lim -lim lim -lim lim]); view(ax, 3);
    xlabel(ax, 'X'); ylabel(ax, 'Y'); zlabel(ax, 'Z'); title_handle = title(ax, 'Snake Robot Animation'); legend(ax, 'show', 'Location', 'northwest');
    
    disp('Press any key to start animation...'); 
    % pause; 
    video_has_been_recorded = true;
    
    % --- Animation Loop Callback Function ---
    function run_animation_loop(~, ~)
        % --- Setup Video Recording ---
        if record_video && ~video_has_been_recorded, v=VideoWriter('snake_animation_insat_v2','Motion JPEG AVI');v.FrameRate=30;open(v);fprintf('\nRecording...\n'); end

        total_time = time_vector(end);
        
        % --- Main Animation Loop (Updated for Locomotion) ---
        for t = 0:(1/60):total_time % Play at 60fps
            % --- Find the current segment ---
            % 'segment_end_times' is [0, T1, T2, T3]
            % 'find' returns the *first* index where t is <= the end time.
            segment_idx = find(t <= segment_end_times(2:end), 1);
            if isempty(segment_idx) % Handle last point
                segment_idx = length(segment_end_times)-1;
            end

            % --- Interpolate Joint Angles ---
            % Get the smooth joint angles for the current time 't'
            q_current = interp1(time_vector, q_trajectory', t)';
            
            % --- Call Generalized FK ---
            % Calculate the robot's world pose using the
            % correct parameters (base pose, mode) for this exact moment.
            [P, T_EE] = forwardKinematics(q_current, link_length, T_base_traj{segment_idx}, base_end_traj{segment_idx});
            if strcmp(base_end_traj{segment_idx},"HEAD_BASE")
                T_store=T_base_traj{segment_idx};
            else
                T_store = T_EE;
            end
            quat = transform2quat(T_base_traj{segment_idx});
            [quat,pos]=transform2quat_pos(T_store);

            % dat = [q_current' quat base_end_traj{segment_idx}]
            if save_execl
                da = coldata2(q_current,quat,pos);
                % da = coldata(q_current,quat,base_end_traj{segment_idx});
                tableda = struct2table(da); 
                writetable(tableda, 'traj_data.xlsx', 'Sheet', '1', 'WriteMode', 'append');
            end
            % --- Update the Plot ---
            % Efficiently update the 'XData', 'YData', 'ZData' of the
            % existing plot handle instead of replotting.
            set(robot_plot, 'XData', P(1,:), 'YData', P(2,:), 'ZData', P(3,:));
            set(title_handle, 'String', sprintf('Time: %.2f s (Mode: %s)', t, base_end_traj{segment_idx}));
            drawnow;
            
            if record_video && ~video_has_been_recorded, writeVideo(v, getframe(fig)); end
        end
        
        % --- Final Frame ---
        set(title_handle, 'String', sprintf('Final Position at Time: %.2f s', total_time));
        if record_video && ~video_has_been_recorded, close(v); fprintf('Recording complete.\n'); video_has_been_recorded = true; end
    end

    % --- Add Replay Button ---
    uicontrol('Parent', fig, 'Style', 'pushbutton', 'String', 'Replay Animation', 'Position', [20 20 120 30], 'Callback', @run_animation_loop);
    % Run the animation the first time
    run_animation_loop();
end

function q = transform2quat(T)
    rotm = T(1:3,1:3);
    q = rotm2quat(rotm);
end

function [q,pos] = transform2quat_pos(T)
    rotm = T(1:3,1:3);
    q = rotm2quat(rotm);
    pos = T(1:3,4);
end

function da = coldata2(q_curr,quat,pos)
    da.a=q_curr(1);
    da.b=q_curr(2);
    da.c=q_curr(3);
    da.d=q_curr(4);
    da.e=q_curr(5);
    da.f=q_curr(6);
    da.g=q_curr(7);
    da.h=q_curr(8);
    da.i=q_curr(9);
    da.j=q_curr(10);
    da.k=q_curr(11);
    da.l=q_curr(12);
    da.m=q_curr(13);
    da.n=q_curr(14);
    da.o=q_curr(15);
    da.p=q_curr(16);

    da.q=quat(1);
    da.r=quat(2);
    da.s=quat(3);
    da.t=quat(4);

    da.u=pos(1);
    da.v=pos(2);
    da.w=pos(3);
    
end

% --- Helper: Get All Points (for Bounding Box) ---
function P_all = getAllPoints(q_traj, link_length, T_base_traj, base_end_traj, seg_times)
    P_all = []; t_vec = linspace(0, seg_times(end), size(q_traj,2));
    for k = 1:size(q_traj, 2)
        % Find segment for this time step
        seg_idx = find(t_vec(k) <= seg_times(2:end), 1);
        if isempty(seg_idx) % Handle last point
             seg_idx = length(seg_times)-1;
        end
        % Calculate pose and append
        [P, ~] = forwardKinematics(q_traj(:,k), link_length, T_base_traj{seg_idx}, base_end_traj{seg_idx});
        P_all = [P_all, P];
    end
end

% --- Helper: Plot Cuboid ---
switch upper(p),case 'X',fs=[f_Ym;f_YM;f_Zm;f_ZM];case 'Y',fs=[f_Xm;f_XM;f_Zm;f_ZM];case 'Z',fs=[f_Xm;f_XM;f_Ym;f_YM];otherwise,fs=[f_Ym;f_YM;f_Zm;f_ZM];end;patch(ax,'Vertices',v,'Faces',fs,'FaceColor','b','FaceAlpha',0.1,'EdgeColor','b','LineStyle','--','DisplayName',dN);end

% --- Helper: Generalized Forward Kinematics (v13+) ---
% (Identical to generate script)
function [P_world, T_ee_world] = forwardKinematics(q, link_length, T_base, base_end)
    n_links = length(q); T_links_in_head_frame=cell(n_links+1,1); T_links_in_head_frame{1}=eye(4); T_current=eye(4);
    for i=1:n_links, if mod(i,2)==1,T_joint=Rz(q(i));else,T_joint=Ry(q(i));end; T_current=T_current*T_joint*Tx(link_length); T_links_in_head_frame{i+1}=T_current; end
    T_tail_in_head_frame = T_links_in_head_frame{n_links+1};
    if strcmpi(base_end, 'HEAD_BASE'), T_world_to_head = T_base; T_ee_world = T_world_to_head * T_tail_in_head_frame;
    elseif strcmpi(base_end, 'TAIL_BASE'), T_world_to_head = T_base * inv(T_tail_in_head_frame); T_ee_world = T_world_to_head;
    else, error('Invalid base_end specified.'); end
    P_world = zeros(3, n_links + 1);
    for i=1:n_links+1, T_link_world=T_world_to_head*T_links_in_head_frame{i}; P_world(:,i)=T_link_world(1:3,4); end
end

% --- Helper: Homogeneous Transforms ---
function T=Rz(t),c=cos(t);s=sin(t);T=[c -s 0 0;s c 0 0;0 0 1 0;0 0 0 1];end; function T=Ry(t),c=cos(t);s=sin(t);T=[c 0 s 0;0 1 0 0;-s 0 c 0;0 0 0 1];end; function T=Tx(d),T=[1 0 0 d;0 1 0 0;0 0 1 0;0 0 0 1];end



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