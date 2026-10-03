% =========================================================================
% Script Name: general_model_fitter.m
% Description: Fits Elastoplastic parameters to experimental total curvature
%              (\kappa) vs residual curvature (\kappa_{res}) data.
%              Features a fully modular architecture supporting N-constitutive
%              models using CMA-ES optimization.
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

% Base directories for inputs and outputs
data_base_dir = fullfile(script_dir, 'Data');
image_base_dir = fullfile(script_dir, 'Images');

% Define which materials to process in this batch
all_base_names = {'PETG', 'BambuPLA1400'};

% Define constitutive models to fit: Voce, GSell_Jonas_Hybrid,
% Orthogonal_Piecewise, CDM
all_model_types = {'GSell_Jonas_Hybrid'};

% Optimization and Parsing Toggles
skip_points = 0;              % Number of points to skip at the end of the CSV data
num_par_workers = 6;          % Maximum number of parallel workers
apply_nadai = true;           % Correct 3PB data using Nadai's correction
use_parallel = false;          % Enable/Disable parallel processing (parfor)

% Visualization Toggles
process_single_file = false;
single_file_name = '';        % Specify filename if process_single_file is true
show_plots = false;           % Toggle to display plots during execution
user_compare_3pb = true;      % Overlay 3PB experimental data on constitutive plots
use_3pb_modulus = false;      % Overwrite theoretical Modulus with 3PB derived Modulus
random_seed = 1;             % Repeatable CMA-ES seeds, one per input file
custom_max_strain = 0.055;    % Maximum strain bound for plotting

% --- MATERIAL PROPERTIES CONFIGURATION ---
% Format: {'Material_Keyword', Half_Thickness_ys (m), E_Modulus (Pa), 'Image_Subfolder', '3PB_CSV_Name'}
% This dynamic configuration table eliminates deep hardcoding in the loops.
material_config = {
    'PETG',      (1.82 / 2) * 1e-3, 1.350333333333333e+09, 'PETG',    'PETG_3pb.csv';
    'BambuPLA1400', (1.40 / 2) * 1e-3, 2.273333333333333e+09, 'BambuPLA1400', 'BambuPLA1400_3pb.csv'
    };

%% 2. BATCH PROCESSING LOOPS

for b_idx = 1:length(all_base_names)
    base_name = all_base_names{b_idx};
    target_folder = fullfile(data_base_dir, base_name);
    if ~isfolder(target_folder); mkdir(target_folder); end

    for m_idx = 1:length(all_model_types)
        model_type = all_model_types{m_idx};

        fprintf('\n======================================================\n');
        fprintf('Processing Material: %s | Model: %s\n', base_name, model_type);
        fprintf('======================================================\n');

        compare_3pb = user_compare_3pb;
        eps_true_all = {};
        sig_true_all = {};

        %% 3. AUTO-CONFIGURE BEAM DIMENSIONS & MATERIAL PROPERTIES

        search_base_folder = image_base_dir;

        % Lookup material properties from config map
        ys = NaN;
        E_hardcoded = NaN;
        subfolder_name = '';
        file_3pb_data = '';

        for cfg_idx = 1:size(material_config, 1)
            if contains(base_name, material_config{cfg_idx, 1}, 'IgnoreCase', true)
                ys = material_config{cfg_idx, 2};
                E_hardcoded = material_config{cfg_idx, 3};
                subfolder_name = material_config{cfg_idx, 4};
                file_3pb_data = fullfile(target_folder, material_config{cfg_idx, 5});
                break;
            end
        end

        if isnan(ys)
            error('Material "%s" not recognized in material_config map.', base_name);
        end

        search_base_folder = fullfile(search_base_folder, subfolder_name);

        %% 4. LOAD AND CORRECT 3PB GROUND TRUTH DATA

        if isfile(file_3pb_data)
            disp('Loading and correcting 3-Point Bending data...');
            data_3pb_raw = readmatrix(file_3pb_data, 'NumHeaderLines', 2);
            num_specimens_3pb = floor(size(data_3pb_raw, 2) / 2);

            E_list = zeros(num_specimens_3pb, 1);
            eps_true_all = cell(num_specimens_3pb, 1);
            sig_true_all = cell(num_specimens_3pb, 1);

            for s = 1:num_specimens_3pb
                eps_raw = data_3pb_raw(:, 2*s - 1);
                sig_raw = data_3pb_raw(:, 2*s);

                valid = ~isnan(eps_raw) & ~isnan(sig_raw);
                if ~any(valid); E_list(s) = NaN; continue; end

                eps_clean = eps_raw(valid) - eps_raw(find(valid, 1));
                sig_app = sig_raw(valid) - sig_raw(find(valid, 1));

                [eps_clean, unique_idx] = unique(eps_clean);
                sig_app = sig_app(unique_idx);

                [~, max_idx] = max(sig_app);
                eps_clean = eps_clean(1:max_idx);
                sig_app = sig_app(1:max_idx);

                if apply_nadai && length(eps_clean) > 15
                    window_size = max(round(length(sig_app) * 0.05), 15);
                    if mod(window_size, 2) == 0; window_size = window_size + 1; end

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

            E_list = E_list(~isnan(E_list));
            if use_3pb_modulus && ~isempty(E_list)
                E = mean(E_list);
                modulus_str = '3PB Modulus';
                modulus_filename_str = '3PB_E';
            else
                E = E_hardcoded;
                modulus_str = 'Hardcoded Modulus';
                modulus_filename_str = 'Hardcoded_E';
            end
        else
            E = E_hardcoded;
            compare_3pb = false;
            modulus_str = 'Hardcoded Modulus';
            modulus_filename_str = 'Hardcoded_E';
            fprintf('WARNING: 3PB data not found at %s. Proceeding without 3PB comparison.\n', file_3pb_data);
        end
        fprintf('Target Folder: %s | E = %.2e Pa\n', target_folder, E);

        %% 5. LOCATE AND FILTER FILES

        files = dir(fullfile(search_base_folder, '**', '*_extracted_curvatures.csv'));
        if isempty(files)
            fprintf('No .csv files found in %s. Skipping material.\n', search_base_folder);
            continue;
        end

        if process_single_file
            idx = find(strcmp({files.name}, single_file_name));
            if isempty(idx)
                fprintf('Single file %s not found. Skipping.\n', single_file_name);
                continue;
            end
            files = files(idx);
        end

        num_files = length(files);
        results_cell = cell(num_files, 1);

        %% 6. SETUP OPTIMIZATION BOUNDS

        if strcmpi(model_type, 'Voce')
            % Parameters: [Yield_Strain (ey), S_Max_Factor, M_Factor]
            xmean = [0.02; 1; 1];
            LBounds = [0.005; 0.01; 0.01];
            UBounds = [0.1; 10; 100];

        elseif strcmpi(model_type, 'GSell_Jonas_Hybrid')
            % Parameters: [Yield_Strain (ey), K_Factor, H_Factor]
            xmean = [0.02; 2; -1];
            LBounds = [0.005; 0.01; -10];
            UBounds = [0.1; 10; 10];

        elseif strcmpi(model_type, 'Orthogonal_Piecewise')
            % Parameters: [Yield_Strain (ey), SP_Factor, Beta_Factor]
            xmean = [0.02; 0.5; 1];
            LBounds = [0.005; 0.01; 0.01];
            UBounds = [0.1; 5; 10];

        elseif strcmpi(model_type, 'CDM')
            % Parameters: [Yield_Strain (ey), C_Factor]
            xmean = [0.02; 1];
            LBounds = [0.005; 0.01];
            UBounds = [0.1; 10];

        else
            error('Model type "%s" not recognized.', model_type);
        end

        insigma = 0.3 * (UBounds - LBounds);

        %% 7. PARALLEL POOL INITIALIZATION
        max_workers = 0; % Honor use_parallel=false even if a pool already exists.
        if use_parallel
            poolobj = gcp('nocreate');
            if isempty(poolobj); poolobj = parpool(min(num_par_workers, num_files)); end
            max_workers = min(num_par_workers, poolobj.NumWorkers);
        end

        %% 8. MAIN PROCESSING LOOP

        parfor (i = 1:num_files, max_workers)
            current_file = files(i).name;
            current_folder = files(i).folder;
            input_file_path = fullfile(current_folder, current_file);
            fprintf('Processing file %d/%d: %s\n', i, num_files, current_file);

            data = readmatrix(input_file_path);
            assert(size(data,2) == 2 && ~isempty(data) && all(isfinite(data(:))), ...
                'Expected two finite columns [total curvature, residual curvature] in %s', input_file_path);
            assert(all(data(:) >= 0) && max(data(:,1)) > 0, ...
                'Curvature magnitudes must be nonnegative, with nonzero total curvature.');
            if size(data, 1) > skip_points
                kappa_list = data(1:end-skip_points, 1);
                kappares_list = data(1:end-skip_points, 2);
            else
                kappa_list = data(:, 1);
                kappares_list = data(:, 2);
            end

            max_exp_strain = max(kappa_list) * ys;
            e_curve = linspace(0, custom_max_strain, 200)';

            % =====================================================================
            % UNIFIED CMA-ES OPTIMIZATION
            % =====================================================================

            opts = cmaes('defaults');
            opts.SaveVariables = 'off';
            opts.LogModulo = 0;
            opts.DispModulo = 0;
            opts.DispFinal = 'off';
            opts.LBounds = LBounds;
            opts.UBounds = UBounds;
            opts.MaxFunEvals = 50000;
            opts.Seed = random_seed + i - 1;

            % Run Hansen's Standard CMA-ES
            [~, ~, ~, ~, ~, bestever] = ...
                cmaes('objective_loss', xmean, insigma, opts, kappa_list, kappares_list, ys, E, model_type);

            x_opt = bestever.x;
            fit_error = bestever.f;

            % Generate Final Reconstructed Stress-Strain Curve
            s_curve = stress_strain_model(e_curve, model_type, x_opt, E);

            % =====================================================================
            % Calculate Predicted k_res for Error Plotting
            % =====================================================================
            kappares_pred_list = zeros(size(kappa_list));
            y_eval = linspace(0, ys, 100);
            e_full = linspace(0, max_exp_strain * 1.5, 100)';
            s_full = stress_strain_model(e_full, model_type, x_opt, E);

            for k_idx = 1:length(kappa_list)
                kappa_val = kappa_list(k_idx);
                strain_profile = kappa_val .* y_eval;
                stress_profile = interp1(e_full, s_full, strain_profile, 'linear', 'extrap');
                moment_integral = trapz(y_eval, stress_profile .* y_eval);
                kappares_pred_list(k_idx) = kappa_val - (3 / (E * ys^3)) * moment_integral;
            end

            % Store Results for Plotting
            res = struct();
            res.name = current_file;
            res.e_curve = e_curve;
            res.s_curve = s_curve;
            res.kappa_exp = kappa_list;
            res.kappares_exp = kappares_list;
            res.kappares_pred = kappares_pred_list;
            res.folder = current_folder;
            results_cell{i} = res;

            % Write Extracted Fit Data to CSV
            [~, name_no_ext, ~] = fileparts(current_file);
            csv_filename = fullfile(target_folder, [name_no_ext, '_', lower(model_type), '_results.csv']);
            fid = fopen(csv_filename, 'w');
            assert(fid ~= -1, 'Cannot create result file: %s', csv_filename);

            fprintf(fid, 'Fit Results (%s)\n', model_type);
            param_str = sprintf('%e,', x_opt);
            fprintf(fid, 'E (Pa),ys (m),fit_error,OptimizedParams\n');
            fprintf(fid, '%e,%e,%e,%s\n\n', E, ys, fit_error, param_str(1:end-1));
            fprintf(fid, 'Stress-Strain Curve\nStrain,Stress (Pa)\n');

            for j = 1:length(e_curve)
                fprintf(fid, '%e,%e\n', e_curve(j), s_curve(j));
            end
            fclose(fid);
        end

        %% 9. POST-PROCESSING AND PLOTTING

        disp('Generating and saving plots...');
        if show_plots; vis_state = 'on'; else; vis_state = 'off'; end

        if ~exist(target_folder, 'dir')
            mkdir(target_folder);
        end

        % --- PLOT 1: INDIVIDUAL PREDICTED VS OBSERVED K_RES ---
        for i = 1:num_files
            if isempty(results_cell{i}); continue; end

            fig_k = figure('Name', ['Fit: ', results_cell{i}.name], 'Color', 'w', 'Visible', vis_state);
            hold on; grid on;

            scatter(results_cell{i}.kappa_exp, results_cell{i}.kappares_exp, 40, 'k', 'filled', 'o', 'DisplayName', 'Experimental');
            plot(results_cell{i}.kappa_exp, results_cell{i}.kappares_pred, 'r-', 'LineWidth', 2.5, 'DisplayName', 'Model Prediction');

            xlabel('Deformation Curvature \kappa (1/m)', 'FontSize', 11);
            ylabel('Residual Curvature \kappa_{res} (1/m)', 'FontSize', 11);
            title(['Springback Fit: ', strrep(results_cell{i}.name, '_', '\_')]);
            legend('Location', 'northwest', 'FontSize', 10);
            set(gca, 'FontSize', 11);

            [~, name_no_ext, ~] = fileparts(results_cell{i}.name);
            saveas(fig_k, fullfile(target_folder, [name_no_ext, '_K_fit.png']));
            if ~show_plots; close(fig_k); end
        end

        % --- PLOT 2: CONSTITUTIVE COMPARISON (IN-SITU OPTIMIZED VS 3PB) ---
        if compare_3pb && exist('eps_true_all', 'var')
            fig_comparison = figure('Name', 'Constitutive Comparison', 'Color', 'w', 'Position', [150, 150, 900, 900], 'Visible', vis_state);
            hold on; grid on;

            for s = 1:length(eps_true_all)
                if ~isempty(eps_true_all{s})
                    plot(eps_true_all{s}, sig_true_all{s} / 1e6, 'Color', [0.6 0.6 0.6], 'LineWidth', 1.5, 'HandleVisibility', 'off');
                end
            end
            plot(NaN, NaN, 'Color', [0.6 0.6 0.6], 'LineWidth', 2, 'DisplayName', 'Corrected 3PB Data');

            max_strain_3pb = 0;
            for s = 1:length(eps_true_all)
                if ~isempty(eps_true_all{s})
                    max_strain_3pb = max(max_strain_3pb, max(eps_true_all{s}));
                end
            end

            if max_strain_3pb > 0
                eps_eval_3pb = linspace(0, max_strain_3pb, 200)';
                sig_3pb_matrix = zeros(length(eps_eval_3pb), length(eps_true_all));
                valid_count = 0;

                for s = 1:length(eps_true_all)
                    if ~isempty(eps_true_all{s})
                        valid_count = valid_count + 1;
                        sig_temp = interp1(eps_true_all{s}, sig_true_all{s}, eps_eval_3pb, 'linear');
                        if any(isnan(sig_temp)); sig_temp(isnan(sig_temp)) = sig_true_all{s}(end); end
                        sig_3pb_matrix(:, valid_count) = sig_temp;
                    end
                end

                if valid_count > 0
                    sig_3pb_mean = mean(sig_3pb_matrix(:, 1:valid_count), 2);
                    plot(eps_eval_3pb, sig_3pb_mean / 1e6, 'k-', 'LineWidth', 3, 'DisplayName', 'Average Corrected 3PB');

                    % Reconstruct 3PB Constitutive Model using unified functions
                    k_max_3pb = max(eps_eval_3pb) / ys;
                    kappa_3pb = linspace(k_max_3pb, 0, 200)';
                    kappares_3pb = zeros(size(kappa_3pb));
                    y_eval_3pb = linspace(-ys, ys, 200);

                    for ki = 1:length(kappa_3pb)
                        k_val = kappa_3pb(ki);
                        if k_val == 0; continue; end
                        strain_val = k_val .* y_eval_3pb;
                        stress_val = sign(strain_val) .* interp1(eps_eval_3pb, sig_3pb_mean, min(abs(strain_val), max(eps_eval_3pb)), 'linear', 'extrap');
                        moment = trapz(y_eval_3pb, stress_val .* y_eval_3pb);
                        kappares_3pb(ki) = k_val - (moment / (E * (2/3)*ys^3));
                    end

                    % Run Standard CMA-ES silently on 3PB Mean Data
                    opts_3pb = cmaes('defaults');
                    opts_3pb.SaveVariables = 'off'; opts_3pb.LogModulo = 0;
                    opts_3pb.DispModulo = 0; opts_3pb.DispFinal = 'off';
                    opts_3pb.LBounds = LBounds; opts_3pb.UBounds = UBounds;
                    opts_3pb.Seed = random_seed;

                    [x_opt_3pb, ~] = cmaes('objective_loss', xmean, insigma, opts_3pb, kappa_3pb, kappares_3pb, ys, E, model_type);
                    sig_3pb_recon = stress_strain_model(eps_eval_3pb, model_type, x_opt_3pb, E);

                    plot(eps_eval_3pb, sig_3pb_recon / 1e6, 'k--', 'LineWidth', 2.5, ...
                        'DisplayName', sprintf('3PB Reconstructed (%s)', strrep(model_type, '_', '\_')));
                end
            end

            % Plot Individual In-Situ Optimized Curves
            colors = hsv(10);
            for i = 1:num_files
                if isempty(results_cell{i}); continue; end
                c = colors(mod(i - 1, 10) + 1, :);
                [~, fname, ~] = fileparts(results_cell{i}.name);
                clean_name = strrep(fname, '_', '\_');

                if i <= 10
                    plot(results_cell{i}.e_curve, results_cell{i}.s_curve / 1e6, 'Color', c, 'LineWidth', 2, 'DisplayName', ['In-Situ: ', clean_name]);
                else
                    plot(results_cell{i}.e_curve, results_cell{i}.s_curve / 1e6, 'Color', c, 'LineWidth', 2, 'HandleVisibility', 'off');
                end
            end

            xlabel('True Strain \epsilon', 'FontSize', 12);
            ylabel('True Stress \sigma (MPa)', 'FontSize', 12);
            title(sprintf('Constitutive Validation: %s - %s (%s)', strrep(base_name, '_', '\_'), strrep(model_type, '_', '\_'), modulus_str));
            legend('Location', 'best', 'FontSize', 10);
            set(gca, 'FontSize', 11);

            file_save_name = sprintf('%s_Comparison_%s_%s.png', base_name, model_type, modulus_filename_str);
            saveas(fig_comparison, fullfile(target_folder, file_save_name));
            if ~show_plots; close(fig_comparison); end
        end
    end


end

fprintf('\nAll batch processing complete.\n');
