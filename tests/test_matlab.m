function test_matlab
% Synthetic unit/interface checks. Requires Curve Fitting Toolbox.
root = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(root,'elastic'), fullfile(root,'plastic'), fullfile(root,'fracture'));
task_temp = tempname;
mkdir(task_temp);
cleanup = onCleanup(@() rmdir(task_temp,'s'));

% Recover known damping/frequency and compare the reconstructed modulus.
t = (0:0.001:0.35)';
zeta = 0.02; omega_n = 500;
y = 10*exp(-zeta*omega_n*t).*cos(omega_n*sqrt(1-zeta^2)*t);
tracking_status = repmat("accepted",size(t));
input = fullfile(task_temp,'tracking.csv');
writetable(table(t,y,tracking_status,'VariableNames', ...
    {'time_s','tip_displacement_px','tracking_status'}), input);
beam = struct('length_m',0.07,'width_m',0.006,'thickness_m',0.0014,'density_kg_m3',1100);
fit_result = response_fit(input,beam,InitialGuess=[0.018,490],ShowPlot=false, ...
    OutputPrefix=string(fullfile(task_temp,'response')));
assert(abs(fit_result.damping_ratio-zeta) < 1e-5);
assert(abs(fit_result.omega_n_rad_s-omega_n) < 1e-3);
expected_E = omega_n^2 * (beam.density_kg_m3*beam.width_m*beam.thickness_m) * ...
    beam.length_m^4 / (1.875^4 * beam.width_m*beam.thickness_m^3/12);
assert(abs(fit_result.effective_modulus_pa/expected_E-1) < 1e-5);

% Constitutive laws should recover the configured elastic slope below yield.
E = 2e9;
types = {'Voce','GSell_Jonas_Hybrid','Orthogonal_Piecewise','CDM'};
params = {[0.02,2,1],[0.02,2,-1],[0.02,2,1],[0.02,1]};
for i=1:numel(types)
    strain = [0,0.001,0.01];
    assert(max(abs(stress_strain_model(strain,types{i},params{i},E)-E*strain)) < 1e-6);
    assert(all(isfinite(stress_strain_model([0.0199,0.02,0.0201,0.05],types{i},params{i},E))));
end
kappa = [1;2;5];
loss = objective_loss([0.02,2,-1],kappa,zeros(size(kappa)),0.0007,E,'GSell_Jonas_Hybrid');
assert(isfinite(loss) && loss < 1e-10); % purely elastic springback is torque-free

% Check original optimizer's modified function-handle entry point.
opts = cmaes('defaults');
opts.SaveVariables='off'; opts.LogModulo=0; opts.DispModulo=0; opts.DispFinal='off';
opts.Seed=1; opts.MaxFunEvals=2000;
[~, value] = cmaes(@(x) sum(x.^2),[1;2],0.5,opts);
assert(value < 1e-8);

% Curvature-to-strain units and calibrated constitutive strength.
curve_file = fullfile(task_temp,'curvature.csv');
writematrix([10,0;20,0;15,0],curve_file);
fracture = fracture_from_curvature(curve_file,0.002,ModelType="GSell_Jonas_Hybrid", ...
    ModelParameters=[0.02,2,-1],ElasticModulus_Pa=E,CalibratedMaxStrain=0.05);
assert(abs(fracture.fracture_strain-0.02) < 1e-12);
assert(abs(fracture.fracture_strength_pa-4e7) < 1);
failed_as_expected = false;
try
    fracture_from_curvature(curve_file,0.002,ModelType="GSell_Jonas_Hybrid", ...
        ModelParameters=[0.02,2,-1],ElasticModulus_Pa=E,CalibratedMaxStrain=0.01);
catch
    failed_as_expected = true;
end
assert(failed_as_expected, 'Strength extrapolation should be rejected.');
fprintf('PASS: MATLAB synthetic response, constitutive laws, moment loss, CMA-ES, and fracture checks.\n');
end
