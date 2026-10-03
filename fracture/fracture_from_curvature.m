function result = fracture_from_curvature(curvature_file, thickness_m, options)
%FRACTURE_FROM_CURVATURE Evaluate manuscript Eqs. (9)-(11).
% Input must describe the FINAL INTACT frame, not an arbitrary plastic load.
% This helper consumes reconstructed curvature; it does not reconstruct it.
% Default column 1 matches plastic/*_extracted_curvatures.csv (no header).
% For legacy curvature.csv, use CurvatureColumn=2 and NumHeaderLines=1.
arguments
    curvature_file (1,1) string
    thickness_m (1,1) double {mustBePositive, mustBeFinite}
    options.CurvatureColumn (1,1) double {mustBeInteger,mustBePositive} = 1
    options.NumHeaderLines (1,1) double {mustBeInteger,mustBeNonnegative} = 0
    options.ModelType (1,1) string = ""
    options.ModelParameters double = []
    options.ElasticModulus_Pa (1,1) double = NaN
    options.CalibratedMaxStrain (1,1) double = NaN
    options.OutputFile (1,1) string = ""
end
data = readmatrix(curvature_file, 'NumHeaderLines', options.NumHeaderLines);
assert(size(data,2) >= options.CurvatureColumn, 'Requested curvature column is absent.');
kappa = data(:,options.CurvatureColumn);
assert(~isempty(kappa) && all(isfinite(kappa)), 'Curvature must be finite and nonempty.');
assert(all(kappa >= 0), 'Provide curvature magnitudes in 1/m, with the bend sign normalized.');
kmax = max(kappa);
strain = kmax * thickness_m / 2;
strength = NaN;
if options.ModelType ~= ""
    validateattributes(options.ElasticModulus_Pa, {'double'}, {'scalar','positive','finite'});
    validateattributes(options.CalibratedMaxStrain, {'double'}, {'scalar','positive','finite'});
    assert(strain <= options.CalibratedMaxStrain, ...
        'Fracture strain exceeds the calibrated constitutive range; strength would be extrapolated.');
    assert(~isempty(options.ModelParameters), 'Provide fitted constitutive parameters.');
    assert(exist('stress_strain_model','file') == 2, 'Add the plastic folder to the MATLAB path.');
    strength = stress_strain_model(strain, options.ModelType, options.ModelParameters, options.ElasticModulus_Pa);
end
result = struct('maximum_curvature_per_m', kmax, 'fracture_strain', strain, 'fracture_strength_pa', strength);
disp(result);
if options.OutputFile ~= ""
    folder = fileparts(options.OutputFile);
    if folder ~= "" && ~isfolder(folder); mkdir(folder); end
    writetable(struct2table(result), options.OutputFile);
end
end
