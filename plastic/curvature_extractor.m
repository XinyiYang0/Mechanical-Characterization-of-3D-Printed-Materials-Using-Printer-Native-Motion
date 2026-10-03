% =========================================================================
% Script Name: curvature_extractor.m
% Description: Extracts curvature from images of slender beams undergoing
%              large deformations. Uses a Physics-Informed Follower-Load
%              Framework to avoid black-box curve fitting.
%
% Author: Victor Riera Naranjo
% ORCID: 0009-0001-8621-0986
% e-mail: vnaranjo6@gatech.edu
% DOI: 10.5281/zenodo.22163290
% Date Created: 2026-08-29
%
% License: BSD 3-Clause License
% Copyright (c) 2026, Victor Riera Naranjo
% All rights reserved.
%
% Redistribution and use in source and binary forms, with or without
% modification, are permitted provided that the following conditions are met:
%
% 1. Redistributions of source code must retain the above copyright notice, this
%    list of conditions and the following disclaimer.
%
% 2. Redistributions in binary form must reproduce the above copyright notice,
%    this list of conditions and the following disclaimer in the documentation
%    and/or other materials provided with the distribution.
%
% 3. Neither the name of the copyright holder nor the names of its
%    contributors may be used to endorse or promote products derived from
%    this software without specific prior written permission.
%
% THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
% AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
% IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
% DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
% FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
% DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
% SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
% CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
% OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
% OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
% =========================================================================

clear; clc; close all;
% Publication copy: portable paths and serial execution by default.
% Scientific equations and optimization bounds retain the supplied implementation.
script_dir = fileparts(mfilename('fullpath'));
addpath(script_dir);

%% 1. USER SETTINGS & CONFIGURATION

base_directory = fullfile(script_dir, 'Images');
new_data_directory = fullfile(script_dir, 'Data');
image_extension = '.png';

% Prescribe the exact physical length of the tracked beam (in meters)
% This is used to scale pixel space back into real physical space.
beam_length = 0.074;

% Degree of the Bernstein polynomial used to fit the residual curvature
% D=5 gives 6 control points. Using too high a degree causes non-physical oscillations.
bernstein_deg = 5;

% Correct 3PB data using Nadai's correction
apply_nadai = true;

% --- Material Thickness Configuration ---
% Map your dataset folder keywords to their respective beam half-thickness (ys) in meters.
% This prevents hardcoding dimensions deep in the logic.
% Format: {'KeywordInFolderName', half_thickness_value}
beam_thickness_map = {
    'PETG',  (1.82 / 2) * 1e-3;
    'Bambu', (1.40 / 2) * 1e-3
    };
fallback_thickness = (2.10 / 2) * 1e-3; % Default if no keyword matches

% --- Visualization Toggles ---
show_plots = false;

save_plots = true;

show_diagnostics = false; % Set to true to watch the optimizer's initial and final states
use_parallel = false;

%% 2. LOCATE MATERIAL DIRECTORIES

% Scan the base directory for folders (each folder represents a material/dataset)
items = dir(base_directory);
material_dirs = items([items.isdir] & ~ismember({items.name}, {'.', '..'}));

if isempty(material_dirs)
    error('No material subdirectories found in %s', base_directory);
end

%% 3. MAIN MATERIAL LOOP

% Initialize parallel processing pool to speed up batch extraction
max_workers = 0; % parfor with zero workers runs on the client.
if use_parallel
    poolobj = gcp('nocreate');
    if isempty(poolobj); poolobj = parpool; end
    max_workers = poolobj.NumWorkers;
end

for m = 1:length(material_dirs)
    mat_name = material_dirs(m).name;
    mat_path = fullfile(base_directory, mat_name);

    fprintf('\n======================================================\n');
    fprintf('Processing Material Group: %s\n', mat_name);
    fprintf('======================================================\n');

    % Dynamically determine half-thickness based on the folder name configuration
    ys = fallback_thickness;
    for t_idx = 1:size(beam_thickness_map, 1)
        if contains(mat_name, beam_thickness_map{t_idx, 1}, 'IgnoreCase', true)
            ys = beam_thickness_map{t_idx, 2};
            break;
        end
    end

    % Look for associated 3-Point Bending (3PB) data to use as physical ground truth
    file_3pb = find_matching_3pb(new_data_directory, mat_name);
    has_3pb = ~isempty(file_3pb) && isfile(file_3pb);

    if has_3pb
        fprintf('Found matching 3PB file: %s\n', file_3pb);
        [k_eval_3pb, k_res_3pb] = compute_3pb_springback(file_3pb, ys, apply_nadai);
    else
        fprintf('No matching 3PB file found for %s.\n', mat_name);
    end

    % Locate all "loaded" image pairs (marked with _1.png)
    files_1 = dir(fullfile(mat_path, '**', ['*_1', image_extension]));
    num_pairs = length(files_1);

    if num_pairs == 0
        fprintf('No image pairs found in %s. Skipping.\n', mat_name);
        continue;
    end

    all_k_tot = cell(num_pairs, 1);
    all_k_res = cell(num_pairs, 1);

    parfor (i = 1:num_pairs, max_workers)
        folder_path = files_1(i).folder;
        img1_name = files_1(i).name;
        img1_path = fullfile(folder_path, img1_name);

        % Construct the name of the residual image pair (_2.png)
        img2_name = strrep(img1_name, ['_1', image_extension], ['_2', image_extension]);
        img2_path = fullfile(folder_path, img2_name);

        base_name = strrep(img1_name, ['_1', image_extension], '');

        if ~isfile(img2_path)
            warning('Missing residual image: %s', img2_path);
            continue;
        end

        % =================================================================
        % CORE PHYSICS ENGINE CALL
        % Extracts raw geometry, normalizes coordinates, builds arc-length splines,
        % enforces physical clamps, and optimizes the unified curvature field.
        % =================================================================
        [s_eval, k_tot, k_res, X1_plot, Y1_plot, X2_plot, Y2_plot, img1_rgb, img2_rgb, X1_raw, Y1_raw, X2_raw, Y2_raw, sload, sy] = ...
            process_physics_beams(img1_path, img2_path, beam_length, bernstein_deg, show_diagnostics);

        % Trim the data: anything past s_load is purely a rigid, straight tail
        % and does not contain valid curvature data. We interpolate to standardize.
        s_common = linspace(0, sload, 500)';
        kappa_total = interp1(s_eval, k_tot, s_common, 'linear', 'extrap');
        kappa_res = interp1(s_eval, k_res, s_common, 'linear', 'extrap');

        % Force residual curvature to absolute zero past the yield boundary (sy)
        % By definition, material past sy only deformed elastically.
        kappa_res(s_common > sy) = 0;

        all_k_tot{i} = kappa_total;
        all_k_res{i} = kappa_res;

        % Save curvature arrays to CSV for external analysis
        csv_filename = fullfile(folder_path, [base_name, '_extracted_curvatures.csv']);
        writematrix([kappa_total, kappa_res], csv_filename);

        if show_plots || save_plots
            if show_plots; vis_state = 'on'; else; vis_state = 'off'; end

            % Figure 1: Image Overlays showing raw extracted pixels vs optimized fit
            fig_img = figure('Name', ['Fit: ', base_name], 'Color', 'w', 'Position', [100, 100, 1200, 500], 'Visible', vis_state);
            subplot(1, 2, 1); imshow(img1_rgb); hold on;
            plot(X1_raw, Y1_raw, 'c.', 'MarkerSize', 5, 'DisplayName', 'Raw Pixels');
            plot(X1_plot, Y1_plot, 'r-', 'LineWidth', 3, 'DisplayName', 'Physics Fit');
            title('Total Bending (Physics Model)');
            legend('Location', 'northeastoutside', 'TextColor', 'k', 'Color', 'w');

            subplot(1, 2, 2); imshow(img2_rgb); hold on;
            plot(X2_raw, Y2_raw, 'c.', 'MarkerSize', 5, 'DisplayName', 'Raw Pixels');
            plot(X2_plot, Y2_plot, 'g-', 'LineWidth', 3, 'DisplayName', 'Physics Fit');
            title('Residual Bending (Physics Model)');
            legend('Location', 'northeastoutside', 'TextColor', 'k', 'Color', 'w');

            % Figure 2: The pure mathematical curvature profiles
            fig_data = figure('Name', ['Data: ', base_name], 'Color', 'w', 'Position', [200, 200, 1000, 400], 'Visible', vis_state);
            subplot(1, 2, 1); hold on; grid on;
            plot(s_common, kappa_total, 'r-', 'LineWidth', 2.5, 'DisplayName', '\kappa (Total)');
            plot(s_common, kappa_res, 'g-', 'LineWidth', 2.5, 'DisplayName', '\kappa_{res} (Residual)');
            xlabel('Arc Length from Root (m)'); ylabel('Curvature (1/m)');
            title('Physics-Informed Profile (Trimmed)'); legend('Location', 'best');

            subplot(1, 2, 2); hold on; grid on;
            plot(kappa_total, kappa_res, 'k-', 'LineWidth', 2.5);
            xlabel('Total Curvature \kappa (1/m)'); ylabel('Residual Curvature \kappa_{res} (1/m)');
            title('Springback Validation (Trimmed)');

            if save_plots
                saveas(fig_img, fullfile(folder_path, [base_name, '_fit_overlay.png']));
                saveas(fig_data, fullfile(folder_path, [base_name, '_curvature_data.png']));
            end
            close(fig_img); close(fig_data);
        end
        fprintf('  -> Successfully processed %s (Load pt: %.1f%%, Yield pt: %.1f%%)\n', base_name, (sload/beam_length)*100, (sy/beam_length)*100);
    end

    % Generate final aggregate plot comparing all samples to the theoretical 3PB curve
    fprintf('Generating Aggregate Plot for %s...\n', mat_name);
    fig_agg = figure('Name', ['Aggregate: ', mat_name], 'Color', 'w', 'Position', [200, 200, 800, 600], 'Visible', 'off');
    hold on; grid on;

    colors = lines(10);
    for i = 1:num_pairs
        if ~isempty(all_k_tot{i})
            color_idx = mod(i - 1, 10) + 1;
            plot(all_k_tot{i}, all_k_res{i}, '-', 'Color', colors(color_idx,:), 'LineWidth', 1.0, 'HandleVisibility', 'off');
        end
    end
    plot(NaN, NaN, '-', 'Color', [0.5 0.5 0.5], 'LineWidth', 1.0, 'DisplayName', 'Extracted Samples (Physics Engine)');

    if has_3pb
        plot(k_eval_3pb, k_res_3pb, 'k-', 'LineWidth', 3.5, 'DisplayName', '3PB Theoretical Prediction');
    end

    xlabel('Total Curvature \kappa (1/m)', 'FontSize', 12);
    ylabel('Residual Curvature \kappa_{res} (1/m)', 'FontSize', 12);
    title(['Springback Validation: ', strrep(mat_name, '_', '\_')]);
    legend('Location', 'northwest', 'FontSize', 11);
    set(gca, 'FontSize', 11);

    saveas(fig_agg, fullfile(mat_path, ['aggregate_comparison_', mat_name, '.png']));
    close(fig_agg);


end
disp('Batch processing complete.');

% =========================================================================
% LOCAL FUNCTION: PHYSICS-INFORMED OPTIMIZER
% Coordinates the extraction, formatting, and optimization of a single beam pair
% =========================================================================
function [s_grid, k_tot_full, k_res_full, X1_plot, Y1_plot, X2_plot, Y2_plot, img1_rgb, img2_rgb, X1_raw, Y1_raw, X2_raw, Y2_raw, s_load, sy] = ...
    process_physics_beams(img1_path, img2_path, L, D, show_diagnostics)

% 1. Image Processing: Extract centerline pixels using HSV color masking
[X1_raw, Y1_raw, img1_rgb] = extract_pixels(img1_path);
[X2_raw, Y2_raw, img2_rgb] = extract_pixels(img2_path);

% 2. Image to Math Coordinate Mapping
Y_max1 = max(Y1_raw); Y_max2 = max(Y2_raw);
x1_raw_math = Y_max1 - Y1_raw; x2_raw_math = Y_max2 - Y2_raw;

% Find the physical root (clamp point) to zero the Y-axis deflection
X1_root = mean(X1_raw(x1_raw_math < 5));
X2_root = mean(X2_raw(x2_raw_math < 5));
y1_raw_math = X1_raw - X1_root;
y2_raw_math = X2_raw - X2_root;

% Force all bends to be treated mathematically as 'positive' deflection
bend_sign = sign(mean(y1_raw_math));
if bend_sign == 0; bend_sign = 1; end
y1_raw_math = y1_raw_math * bend_sign;
y2_raw_math = y2_raw_math * bend_sign;

% 3. SORT PIXELS ROBUSTLY FROM ROOT TO TIP
[x1_raw_math, y1_raw_math] = robust_pixel_sort(x1_raw_math, y1_raw_math);
[x2_raw_math, y2_raw_math] = robust_pixel_sort(x2_raw_math, y2_raw_math);

% 4. Physical Scaling (Pixels to Meters)
unique_x = unique(x2_raw_math); mean_y = zeros(size(unique_x));
for k = 1:length(unique_x)
    mean_y(k) = mean(y2_raw_math(x2_raw_math == unique_x(k)));
end
s_px_approx = sum(sqrt(diff(unique_x).^2 + diff(mean_y).^2));
px_per_m = s_px_approx / L;

x1_data = x1_raw_math ./ px_per_m; y1_data = y1_raw_math ./ px_per_m;
x2_data = x2_raw_math ./ px_per_m; y2_data = y2_raw_math ./ px_per_m;

% 5. Build Analytical Geometric Splines
s_grid = linspace(0, L, 500)';
[x1_spl, y1_spl, th1_spl] = build_spline(x1_data, y1_data, s_grid, L);
[x2_spl, y2_spl, th2_spl] = build_spline(x2_data, y2_data, s_grid, L);

% --- ENFORCE PERFECT ROOT CLAMP (y'(0) = 0) ---
th0_1 = 0; x0_1 = x1_spl(1); y0_1 = y1_spl(1);
[x1_spl, y1_spl, th1_spl] = align_to_root(x1_spl, y1_spl, th1_spl);

th0_2 = 0; x0_2 = x2_spl(1); y0_2 = y2_spl(1);
[x2_spl, y2_spl, th2_spl] = align_to_root(x2_spl, y2_spl, th2_spl);

% 6. Optimizer Setup
% p0 = [s_load, s_y, lambda_scaled, delta_th, deltas]
s_load_guess = 0.85 * L;
s_y_guess = 0.65 * L;
num_R_deltas = D - 1;
lambda_scale = 1000;
lambda_guess_scaled = 1.0;
delta_th_guess = 0.0;
alpha_guess = 0.0;

p0 = [s_load_guess, s_y_guess, lambda_guess_scaled, delta_th_guess, alpha_guess, ones(1, num_R_deltas)*0.1];

% 7. Generate Adaptive Objective Normalizers
[~, ~, ~, ~, ~, ~, ~, ~, err_struct] = physics_loss(p0, s_grid, x1_spl, y1_spl, th1_spl, x2_spl, y2_spl, th2_spl, D, lambda_scale, []);
norm_factors.pos  = max(err_struct.pos,  1e-10);
norm_factors.th   = max(err_struct.th,   1e-10);
norm_factors.bc   = max(err_struct.bc,   1e-10);
norm_factors.tail = max(err_struct.tail, 1e-10);

if show_diagnostics
    plot_diagnostics(p0, s_grid, x1_spl, y1_spl, th1_spl, x2_spl, y2_spl, th2_spl, D, lambda_scale, 'INITIAL GUESS STATE', norm_factors);
end

obj_fun = @(p) physics_loss(p, s_grid, x1_spl, y1_spl, th1_spl, x2_spl, y2_spl, th2_spl, D, lambda_scale, norm_factors);
options = optimset('Display','none','MaxFunEvals', 10000, 'MaxIter', 10000);

% Linear Constraints (A*x <= b)
A = zeros(4, length(p0)); b = zeros(4, 1);
A(1,1) = 1;  b(1) = 0.99*L;
A(2,1) = -1;
A(3,1) = -1; A(3,2) = 1;
A(4,2) = -1;

% Hard parameter bounds
lb = [0.2* L, 0.05* L, 0, -0.1, -0.1, zeros(1, num_R_deltas)];
ub = [L, L, Inf, 0.1, 2.0, Inf(1, num_R_deltas)];

% Execute optimization
p_opt = fmincon(obj_fun, p0, A, b, [], [], lb, ub, [], options);

if show_diagnostics
    plot_diagnostics(p_opt, s_grid, x1_spl, y1_spl, th1_spl, x2_spl, y2_spl, th2_spl, D, lambda_scale, 'FINAL OPTIMIZED STATE', norm_factors);
end

% 8. Reconstruct the Final Analytical Curves using optimal parameters
[~, k_tot_full, k_res_full, x_tot, y_tot, x_res, y_res] = ...
    physics_loss(p_opt, s_grid, x1_spl, y1_spl, th1_spl, x2_spl, y2_spl, th2_spl, D, lambda_scale, norm_factors);

s_load = p_opt(1);
sy = p_opt(2);

% 9. Un-align Math Arrays back to raw Math Frame
R_inv1 = [cos(th0_1), -sin(th0_1); sin(th0_1), cos(th0_1)];
pts1 = R_inv1 * [x_tot, y_tot]';
x_tot_unaligned = pts1(1,:)' + x0_1;
y_tot_unaligned = pts1(2,:)' + y0_1;

R_inv2 = [cos(th0_2), -sin(th0_2); sin(th0_2), cos(th0_2)];
pts2 = R_inv2 * [x_res, y_res]';
x_res_unaligned = pts2(1,:)' + x0_2;
y_res_unaligned = pts2(2,:)' + y0_2;

% 10. Map the Math Arrays perfectly back into Image Coordinates for plotting
Y1_plot = Y_max1 - (x_tot_unaligned .* px_per_m);
Y2_plot = Y_max2 - (x_res_unaligned .* px_per_m);
X1_plot = X1_root + bend_sign .* (y_tot_unaligned .* px_per_m);
X2_plot = X2_root + bend_sign .* (y_res_unaligned .* px_per_m);


end

% =========================================================================
% LOCAL FUNCTION: HSV Pixel Extraction
% =========================================================================
function [X, Y, img] = extract_pixels(path)
img = imread(path); hsv = rgb2hsv(img);
mask = (hsv(:,:,1) >= 10/180) & (hsv(:,:,1) <= 35/180) & ...
    (hsv(:,:,2) >= 60/255) & (hsv(:,:,3) >= 80/255);
mask = bwareafilt(mask, 1); % Keep only the largest connected component
[Y, X] = find(mask);
if isempty(X); error('No marker found.'); end
end

% =========================================================================
% LOCAL FUNCTION: Robust Pixel Sorting (Nearest Neighbor)
% =========================================================================
function [x_sort, y_sort] = robust_pixel_sort(x, y)
[~, idx] = min(x);
unvisited = true(length(x), 1);
x_sort = zeros(size(x));
y_sort = zeros(size(y));

curr_x = x(idx);
curr_y = y(idx);
unvisited(idx) = false;
x_sort(1) = curr_x;
y_sort(1) = curr_y;

for i = 2:length(x)
    rem_idx = find(unvisited);
    if isempty(rem_idx); break; end

    dist2 = (x(rem_idx) - curr_x).^2 + (y(rem_idx) - curr_y).^2;
    [~, min_idx] = min(dist2);

    next_idx = rem_idx(min_idx);
    curr_x = x(next_idx);
    curr_y = y(next_idx);

    x_sort(i) = curr_x;
    y_sort(i) = curr_y;
    unvisited(next_idx) = false;
end


end

% =========================================================================
% LOCAL FUNCTION: Build Arc-Length Parameterized Spline
% =========================================================================
function [x_spl, y_spl, th_spl] = build_spline(x_data, y_data, s_grid, L)
[unq_x, ~, ic] = unique(x_data, 'stable');
mean_y = zeros(size(unq_x));
for i = 1:length(unq_x)
    mean_y(i) = mean(y_data(ic == i));
end
x_data = unq_x;
y_data = mean_y;

window = max(round(length(x_data) * 0.08), 5);
x_sm = smoothdata(x_data, 'sgolay', window);
y_sm = smoothdata(y_data, 'sgolay', window);

ds = sqrt(diff(x_sm).^2 + diff(y_sm).^2);
s_data = [0; cumsum(ds)];

[s_data, unq_s_idx] = unique(s_data, 'stable');
x_sm = x_sm(unq_s_idx);
y_sm = y_sm(unq_s_idx);

s_data = (s_data / max(s_data)) * L;

x_spl = interp1(s_data, x_sm, s_grid, 'spline', 'extrap');
y_spl = interp1(s_data, y_sm, s_grid, 'spline', 'extrap');

dx = gradient(x_spl, s_grid); dy = gradient(y_spl, s_grid);
th_spl = atan2(dy, dx);


end

% =========================================================================
% LOCAL FUNCTION: Align Spline to Perfect Cantilever Root
% =========================================================================
function [x_out, y_out, th_out] = align_to_root(x_in, y_in, th_in)
x_shift = x_in - x_in(1);
y_shift = y_in - y_in(1);

th0 = 0;

R = [cos(-th0), -sin(-th0); sin(-th0), cos(-th0)];
coords = R * [x_shift, y_shift]';

x_out = coords(1,:)';
y_out = coords(2,:)';
th_out = th_in - th0;
th_out(1) = 0;


end

% =========================================================================
% LOCAL FUNCTION: Physics-Informed Objective Function
% =========================================================================
function [err, k_tot, k_res, x_rec_tot, y_rec_tot, x_rec_res, y_rec_res, tail_data, err_struct] = ...
    physics_loss(p, s_grid, x1_spl, y1_spl, th1_spl, x2_spl, y2_spl, th2_spl, D, lambda_scale, norm_factors)

s_load = p(1);
sy = p(2);
lambda = p(3) * lambda_scale;
delta_th = p(4);
alpha = p(5);

deltas = p(6:end);
R = zeros(1, D+1);
curr_R = 0;
for i = 1:length(deltas)
    curr_R = curr_R + deltas(i)^2;
    R(D - 1 - i + 1) = curr_R;
end

L = s_grid(end);
valid_domain = s_grid <= 0.95 * L;

tail1 = s_grid >= s_load & valid_domain;
tail2 = s_grid >= sy & valid_domain;

if sum(tail1) < 2
    th1_load = th1_spl(end);
    tail_data.x1_fit = x1_spl(end);
    tail_data.y1_fit = y1_spl(end);
else
    px1 = polyfit(x1_spl(tail1), y1_spl(tail1), 1);
    th1_load = atan2(px1(1), 1);
    tail_data.x1_fit = x1_spl(tail1);
    tail_data.y1_fit = polyval(px1, x1_spl(tail1));
end

if sum(tail2) < 2
    th2_y = th2_spl(end);
    tail_data.x2_fit = x2_spl(end);
    tail_data.y2_fit = y2_spl(end);
else
    px2 = polyfit(x2_spl(tail2), y2_spl(tail2), 1);
    th2_y = atan2(px2(1), 1);
    tail_data.x2_fit = x2_spl(tail2);
    tail_data.y2_fit = polyval(px2, x2_spl(tail2));
end

x1_load = interp1(s_grid, x1_spl, s_load, 'linear', 'extrap');
y1_load = interp1(s_grid, y1_spl, s_load, 'linear', 'extrap');

k_res = zeros(size(s_grid));
mask_res = s_grid <= sy;
eta_res = s_grid(mask_res) / sy;
for i = 0:D
    if R(i+1) > 0
        k_res(mask_res) = k_res(mask_res) + R(i+1) * nchoosek(D, i) .* ((1-eta_res).^(D-i)) .* (eta_res.^i);
    end
end

k_el = zeros(size(s_grid));
mask_el = s_grid <= s_load;
x1_el = x1_spl(mask_el); y1_el = y1_spl(mask_el);

applied_angle = th1_load + delta_th;
k_el(mask_el) = lambda .* ( (x1_load - x1_el).*cos(applied_angle) + (y1_load - y1_el).*sin(applied_angle) );

k_tot = k_res + k_el + (alpha .* k_res);

th_tot = cumtrapz(s_grid, k_tot);
x_rec_tot = cumtrapz(s_grid, cos(th_tot));
y_rec_tot = cumtrapz(s_grid, sin(th_tot));

th_res = cumtrapz(s_grid, k_res);
x_rec_res = cumtrapz(s_grid, cos(th_res));
y_rec_res = cumtrapz(s_grid, sin(th_res));

idx_load = find(s_grid >= s_load, 1);
if ~isempty(idx_load)
    th_b = th_tot(idx_load); x_b = x_rec_tot(idx_load); y_b = y_rec_tot(idx_load);
    ds_tail = s_grid(tail1) - s_load;
    th_tot(tail1) = th_b;
    x_rec_tot(tail1) = x_b + ds_tail * cos(th_b);
    y_rec_tot(tail1) = y_b + ds_tail * sin(th_b);
end

idx_y = find(s_grid >= sy, 1);
if ~isempty(idx_y)
    th_b = th_res(idx_y); x_b = x_rec_res(idx_y); y_b = y_rec_res(idx_y);
    ds_tail = s_grid(tail2) - sy;
    th_res(tail2) = th_b;
    x_rec_res(tail2) = x_b + ds_tail * cos(th_b);
    y_rec_res(tail2) = y_b + ds_tail * sin(th_b);
end

err_pos_tot = mean((x_rec_tot(valid_domain) - x1_spl(valid_domain)).^2 + (y_rec_tot(valid_domain) - y1_spl(valid_domain)).^2);
err_pos_res = mean((x_rec_res(valid_domain) - x2_spl(valid_domain)).^2 + (y_rec_res(valid_domain) - y2_spl(valid_domain)).^2);

err_th_tot = mean((th_tot(valid_domain) - th1_spl(valid_domain)).^2);
err_th_res = mean((th_res(valid_domain) - th2_spl(valid_domain)).^2);

err_bc_tot = (th_tot(idx_load) - th1_load)^2;
err_bc_res = (th_res(idx_y) - th2_y)^2;

err_tail_tot = mean((x_rec_tot(tail1) - tail_data.x1_fit).^2 + (y_rec_tot(tail1) - tail_data.y1_fit).^2);
if isnan(err_tail_tot); err_tail_tot = 0; end

err_tail_res = mean((x_rec_res(tail2) - tail_data.x2_fit).^2 + (y_rec_res(tail2) - tail_data.y2_fit).^2);
if isnan(err_tail_res); err_tail_res = 0; end

min_sep = 0.05 * L;
err_sy_sep = max(0, min_sep - (s_load - sy))^2;

WEIGHTS = [2.276620648658670e+02, 18.199037976892161, 26.350373423391950, ...
    76.812883278727412, 93.830366650674634, 30.000155524969863, ...
    51.816862838040478, 59.399762185843024, 12.274488565366942];

W_LOAD = WEIGHTS(1);
W_RES  = WEIGHTS(2);

err_struct.pos  = W_LOAD * err_pos_tot  + W_RES * err_pos_res;
err_struct.th   = W_LOAD * err_th_tot   + W_RES * err_th_res;
err_struct.bc   = W_LOAD * err_bc_tot   + W_RES * err_bc_res;
err_struct.tail = W_LOAD * err_tail_tot + W_RES * err_tail_res;

if isempty(norm_factors)
    err = 0;
else
    w_pos  = WEIGHTS(3);
    w_th   = WEIGHTS(4);
    w_bc   = WEIGHTS(5);
    w_tail = WEIGHTS(6);

    w_fric = WEIGHTS(7);
    err_fric = (delta_th / 0.1)^2;

    w_sep = WEIGHTS(8) * 1e6;
    w_alpha = WEIGHTS(9) * 1e6;
    err_alpha = (100 * alpha^2) / (1 + 100 * alpha^2);

    err = w_pos  * (err_struct.pos  / norm_factors.pos) + ...
        w_th   * (err_struct.th   / norm_factors.th) + ...
        w_bc   * (err_struct.bc   / norm_factors.bc) + ...
        w_tail * (err_struct.tail / norm_factors.tail) + ...
        w_fric * err_fric + ...
        w_sep  * err_sy_sep + ...
        w_alpha * err_alpha;
end


end

% =========================================================================
% LOCAL FUNCTION: Diagnostic Plotter
% =========================================================================
function plot_diagnostics(p, s_grid, x1_spl, y1_spl, th1_spl, x2_spl, y2_spl, th2_spl, D, lambda_scale, title_str, norm_factors)

[~, k_tot, k_res, x_rec_tot, y_rec_tot, x_rec_res, y_rec_res, tail_data] = ...
    physics_loss(p, s_grid, x1_spl, y1_spl, th1_spl, x2_spl, y2_spl, th2_spl, D, lambda_scale, norm_factors);

s_load = p(1); sy = p(2);
tail1 = s_grid >= s_load; tail2 = s_grid >= sy;

fig = figure('Name', title_str, 'Color', 'w', 'Position', [100, 100, 1200, 800]);

subplot(2, 2, 1); hold on; grid on;
plot(x1_spl, y1_spl, 'k-', 'LineWidth', 4, 'DisplayName', 'Target Spline (Loaded)');
plot(x_rec_tot, y_rec_tot, 'r--', 'LineWidth', 2, 'DisplayName', 'Reconstruction');
plot(tail_data.x1_fit, tail_data.y1_fit, 'c-', 'LineWidth', 2, 'DisplayName', 'Dynamic Tail Fit');
plot(x1_spl(tail1), y1_spl(tail1), 'g.', 'DisplayName', 'Tail Points Used');
scatter(x_rec_tot(find(tail1,1)), y_rec_tot(find(tail1,1)), 100, 'rp', 'filled', 'DisplayName', 's_{load}');
scatter(x1_spl(1), y1_spl(1), 100, 'ks', 'filled', 'DisplayName', 'Root');
title([title_str, ' - Loaded Shape']); xlabel('X (m)'); ylabel('Y (m)'); legend('Location', 'best'); axis equal;

subplot(2, 2, 2); hold on; grid on;
plot(x2_spl, y2_spl, 'k-', 'LineWidth', 4, 'DisplayName', 'Target Spline (Residual)');
plot(x_rec_res, y_rec_res, 'r--', 'LineWidth', 2, 'DisplayName', 'Reconstruction');
plot(tail_data.x2_fit, tail_data.y2_fit, 'c-', 'LineWidth', 2, 'DisplayName', 'Dynamic Tail Fit');
plot(x2_spl(tail2), y2_spl(tail2), 'g.', 'DisplayName', 'Tail Points Used');
scatter(x_rec_res(find(tail2,1)), y_rec_res(find(tail2,1)), 100, 'rp', 'filled', 'DisplayName', 's_y');
scatter(x2_spl(1), y2_spl(1), 100, 'ks', 'filled', 'DisplayName', 'Root');
title([title_str, ' - Residual Shape']); xlabel('X (m)'); ylabel('Y (m)'); legend('Location', 'best'); axis equal;

subplot(2, 2, [3 4]); hold on; grid on;
plot(s_grid, k_tot, 'r-', 'LineWidth', 2, 'DisplayName', '\kappa_{tot}');
plot(s_grid, k_res, 'g-', 'LineWidth', 2, 'DisplayName', '\kappa_{res}');
plot(s_grid, k_tot - k_res, 'b--', 'LineWidth', 2, 'DisplayName', '\kappa_{el} (Derived)');
xline(s_load, 'r--', 's_{load}'); xline(sy, 'g--', 's_y');
title([title_str, ' - Curvatures']); xlabel('s (m)'); ylabel('Curvature (1/m)'); legend('Location', 'best');

drawnow; pause(1);


end

% =========================================================================
% LOCAL FUNCTION: 3PB THEORETICAL SPRINGBACK
% =========================================================================
function [k_eval_array, k_res_pred] = compute_3pb_springback(file_3pb, ys, apply_nadai)
data_3pb_raw = readmatrix(file_3pb, 'NumHeaderLines', 2);
num_specimens_3pb = floor(size(data_3pb_raw, 2) / 2);

E_list = zeros(num_specimens_3pb, 1);
eps_true_all = cell(num_specimens_3pb, 1);
sig_true_all = cell(num_specimens_3pb, 1);

for s = 1:num_specimens_3pb
    eps_raw = data_3pb_raw(:, 2*s - 1);
    sig_raw = data_3pb_raw(:, 2*s);

    valid = ~isnan(eps_raw) & ~isnan(sig_raw);
    if ~any(valid)
        E_list(s) = NaN;
        continue;
    end

    eps_clean = eps_raw(valid) - eps_raw(find(valid, 1));
    sig_app = sig_raw(valid) - sig_raw(find(valid, 1));

    [eps_clean, unique_idx] = unique(eps_clean);
    sig_app = sig_app(unique_idx);

    [~, max_idx] = max(sig_app);
    eps_clean = eps_clean(1:max_idx);
    sig_app = sig_app(1:max_idx);

    if apply_nadai && length(eps_clean) > 15
        window_size = max(round(length(sig_app) * 0.05), 15);
        if mod(window_size, 2) == 0
            window_size = window_size + 1;
        end

        sig_smoothed = smoothdata(sig_app, 'sgolay', window_size);
        dsig_deps = gradient(sig_smoothed, eps_clean);
        dsig_deps_smooth = smoothdata(dsig_deps, 'sgolay', window_size);

        sig_true = (2 .* sig_app + eps_clean .* dsig_deps_smooth) / 3;
        sig_true = smoothdata(sig_true, 'sgolay', window_size);
        sig_true(sig_true < 0) = 0;
    else
        sig_true = sig_app;
    end

    eps_true_all{s} = eps_clean;
    sig_true_all{s} = sig_true;

    linear_region = sig_true > 0.1 * max(sig_true) & sig_true < 0.4 * max(sig_true);
    if sum(linear_region) > 5
        p = polyfit(eps_clean(linear_region), sig_true(linear_region), 1);
        E_list(s) = p(1);
    else
        E_list(s) = NaN;
    end
end
E_3pb = mean(E_list);

max_strain_3pb = 0;
for s = 1:length(eps_true_all)
    max_strain_3pb = max(max_strain_3pb, max(eps_true_all{s}));
end

eps_eval_3pb = linspace(0, max_strain_3pb, 200)';
sig_3pb_matrix = zeros(length(eps_eval_3pb), length(eps_true_all));
valid_count = 0;

for s = 1:length(eps_true_all)
    if ~isempty(eps_true_all{s})
        valid_count = valid_count + 1;
        sig_temp = interp1(eps_true_all{s}, sig_true_all{s}, eps_eval_3pb, 'linear');
        if any(isnan(sig_temp))
            sig_temp(isnan(sig_temp)) = sig_true_all{s}(end);
        end
        sig_3pb_matrix(:, valid_count) = sig_temp;
    end
end

sig_3pb_mean = mean(sig_3pb_matrix(:, 1:valid_count), 2);
% plot(eps_eval_3pb, sig_3pb_mean / 1e6, 'k-', 'LineWidth', 3, 'DisplayName', 'Average Corrected 3PB');

k_max_3pb = max(eps_eval_3pb) / ys;
k_eval_array = linspace(0, k_max_3pb, 200)';
k_res_pred = zeros(size(k_eval_array));
y_eval_3pb = linspace(-ys, ys, 200);

for ki = 1:length(k_eval_array)
    k_val = k_eval_array(ki);
    if k_val == 0; continue; end

    strain_val = k_val .* y_eval_3pb;
    stress_val = sign(strain_val) .* interp1(eps_eval_3pb, sig_3pb_mean, min(abs(strain_val), max(eps_eval_3pb)), 'linear', 'extrap');
    moment = trapz(y_eval_3pb, stress_val .* y_eval_3pb);
    k_res_pred(ki) = k_val - (moment / (E_3pb * (2/3)*ys^3));
end


end

% =========================================================================
% LOCAL FUNCTION: Match 3PB File Dynamically
% =========================================================================
function file_3pb = find_matching_3pb(search_dir, mat_name)
% Searches for any CSV file containing '3pb' in the specified directory.
% It checks if the filename shares keywords with the material folder name.

files = dir(fullfile(search_dir, '**', '*3pb*.csv'));
file_3pb = '';

% Break folder name into potential keyword parts (handling underscores if used)
keywords = split(mat_name, '_');

for i = 1:length(files)
    fname = files(i).name;
    match = false;

    for k = 1:length(keywords)
        % If the CSV file shares a keyword with the folder, consider it a match
        if ~isempty(keywords{k}) && contains(fname, keywords{k}, 'IgnoreCase', true)
            match = true;
            break;
        end
    end

    if match
        file_3pb = fullfile(files(i).folder, fname);
        return;
    end
end


end
