% This script loads a pre-computed trajectory and calculates the joint torques
% required to execute the motion using the Recursive Newton-Euler Algorithm (RNEA).
% It plots the total torques and the torques without the effect of gravity.

clear; clc; close all;

% =========================================================================
% 1. LOAD TRAJECTORY DATA AND DEFINE PHYSICAL PARAMETERS
% =========================================================================

% --- Load Data ---
% You can swap this out for whichever file contains your 'q' matrix
% load('my_data_2.mat'); 
% q_traj = my_matrix(:, 1:15); % Example: Ensure this is (n_links x timesteps)

% % --- OR Use the standard loader ---
% data_filename = 'trajectory_data_v15_gym.mat';
% if exist(data_filename, 'file')
%     load(data_filename);
%     % ASSIGN YOUR TRAJECTORY HERE
%     if exist('my_matrix', 'var')
%         q_traj = my_matrix; 
%     elseif exist('full_q_opt', 'var')
%         q_traj = full_q_opt;
%     else
%         error('No trajectory variable found (q_traj, my_matrix, or full_q_opt).');
%     end
% else
%      % If file missing, generate dummy data or error
%      error('Data file not found.');
% end

load('data_torq_limts.mat')
my_matrix = my_matrix(:,1:15);
q_traj=my_matrix';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% n_interp_points = 5; % We are interpolating the first 5 points
% n_intra_points = 20;
% q_startup = q_traj(:, 1:n_interp_points);
% q_remainder = q_traj(:, n_interp_points+1:end);
% t_original = 1:n_interp_points;
% t_interp = linspace(1, n_interp_points, (n_interp_points-1)*n_intra_points + 1);
% q_startup_interp = spline(t_original, q_startup, t_interp);
% q_traj = [q_startup_interp, q_remainder];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%








% --- Define Time Step (CRITICAL) ---
% If your data file has a time vector, calculate dt from it.
% Otherwise, manually set the time step (e.g., 0.01s).
if exist('full_time_vector', 'var')
    dt = full_time_vector(2) - full_time_vector(1);
    fprintf('Detected time step from data: dt = %.4f s\n', dt);
else
    dt = 0.05; 
    fprintf('Using manual time step: dt = %.4f s\n', dt);
    num_steps = size(q_traj, 2);
    full_time_vector = 0:dt:(num_steps-1)*dt;
end

% --- Calculate q_dot (Velocity) and q_ddot (Acceleration) ---
% We use the 'gradient' function which preserves vector length (unlike diff)
[n_links, num_timesteps] = size(q_traj);
q_dot_calc = zeros(n_links, num_timesteps);
q_ddot_calc = zeros(n_links, num_timesteps);

for i = 1:n_links
    % First derivative (Velocity)
    q_dot_calc(i, :) = gradient(q_traj(i, :), dt);
    % Second derivative (Acceleration)
    q_ddot_calc(i, :) = gradient(q_dot_calc(i, :), dt);
end

% --- Define Physical Properties ---
link_mass = 0.100; % Mass of each link in kg
gravity_vec = [0; 0; -9.81]; % Gravity vector in m/s^2

% --- Unit Conversion ---
% Ensure q_traj is in Radians. If in degrees, uncomment below:
% q_traj = deg2rad(q_traj); 

% Ensure link length is defined. 
if ~exist('link_length', 'var')
    link_length = 5.06; % Default to 14cm if missing
end
link_length_m = link_length / 100; % Convert cm to m

fprintf('Analyzing dynamics with link mass = %.3f kg and link length = %.4f m\n', link_mass, link_length_m);

% =========================================================================
% 2. CALCULATE TORQUES FOR EACH TIME STEP
% =========================================================================

torques_total = zeros(n_links, num_timesteps);
torques_no_gravity = zeros(n_links, num_timesteps);
torques_gravity_only = zeros(n_links, num_timesteps);

fprintf('Calculating inverse dynamics for %d time steps...\n', num_timesteps);

tic;
for k = 1:num_timesteps
    % Extract state and control for the current time step
    % NOW USING CALCULATED VALUES
    q = q_traj(:, k);
    q_dot = q_dot_calc(:, k);
    q_ddot = q_ddot_calc(:, k); 
    
    % Calculate total torque (Dynamics + Gravity)
    torques_total(:, k) = inverseDynamicsRNEA(q, q_dot, q_ddot, n_links, link_length_m, link_mass, gravity_vec);
    
    % Calculate torque due to gravity only (set velocities/accel to 0)
    torques_gravity_only(:, k) = inverseDynamicsRNEA(q, zeros(n_links,1), zeros(n_links,1), n_links, link_length_m, link_mass, gravity_vec);
    
    % Calculate torque due to motion dynamics only
    torques_no_gravity(:, k) = torques_total(:, k) - torques_gravity_only(:, k);
end
calc_time = toc;
fprintf('Dynamics calculation finished in %.2f seconds.\n', calc_time);

% =========================================================================
% 3. PLOT THE RESULTS
% =========================================================================
figure('Name', 'Joint Torque Analysis', 'NumberTitle', 'off', 'WindowState', 'maximized');
joint_legend = arrayfun(@(d) sprintf('Joint %d', d), 1:n_links, 'UniformOutput', false);

% --- Plot 1: Total Torques (With Gravity) ---
subplot(2, 1, 1);
plot(full_time_vector, torques_total, 'LineWidth', 1.5);
title('Total Joint Torques (Motion + Gravity)');
xlabel('Time (s)');
ylabel('Torque (Nm)');
grid on;
legend(joint_legend, 'Location', 'eastoutside');

% --- Plot 2: Torques without Gravity ---
subplot(2, 1, 2);
plot(full_time_vector, torques_no_gravity, 'LineWidth', 1.5);
title('Dynamic Joint Torques (Motion Only)');
xlabel('Time (s)');
ylabel('Torque (Nm)');
grid on;
legend(joint_legend, 'Location', 'eastoutside');

% =========================================================================
% LOCAL FUNCTIONS
% =========================================================================
% (Paste your original inverseDynamicsRNEA, Rz, and Ry functions here)
% They remain exactly the same.
function tau = inverseDynamicsRNEA(q, q_dot, q_ddot, n_links, L, m, g)
    % ... [Your original RNEA code] ...
    % Only including the header to save space, keep the full function from your script.
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
            R_world_to_body = Rz(q(i));
            joint_axis = z_axis;
        else 
            R_world_to_body = Ry(q(i));
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
            if mod(i+1, 2) == 1, R_body_to_world_next = Rz(q(i+1))'; else, R_body_to_world_next = Ry(q(i+1))'; end
            f(:,i) = R_body_to_world_next * f(:,i+1) + F;
            n(:,i) = N + R_body_to_world_next * n(:,i+1) + cross(CoM, F) + cross(p_link, R_body_to_world_next*f(:,i+1));
        end
        
        if mod(i, 2) == 1
            joint_axis = z_axis;
        else
            joint_axis = y_axis;
        end
        
        tau(i) = n(:,i)' * joint_axis;
        if i<5
            tau(i)=0;
        end
    end
end

function Rot = Rz(theta)
    c = cos(theta); s = sin(theta);
    Rot = [c, -s, 0; s, c, 0; 0, 0, 1];
end
function Rot = Ry(theta)
    c = cos(theta); s = sin(theta);
    Rot = [c, 0, s; 0, 1, 0; -s, 0, c];
end