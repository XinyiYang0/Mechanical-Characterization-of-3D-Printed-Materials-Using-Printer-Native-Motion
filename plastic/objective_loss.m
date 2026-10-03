% =========================================================================
% Script Name: objective_loss.m
% Description: Unified Objective Loss evaluating Equilibrium Torque Violation.
%              Calculates the mathematical penalty based on how far the
%              beam is from true static physical equilibrium.
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

function loss = objective_loss(x, kappa_list, kappares_list, ys, E, model_type)

% 1. Pre-compute lookup table for the stress curve
emax = max(kappa_list) * ys;
e_list = linspace(0, emax * 1.5, 50)';
s_list = stress_strain_model(e_list, model_type, x, E);

% Secondary sanity check: prevent numerical explosion 
% (e.g. stress exceeding 5x Young's Modulus)
if any(isnan(s_list)) || any(isinf(s_list)) || max(abs(s_list)) > 5 * E
    loss = 1e6;
    return;
end

total_loss_torque = 0;
y_list = linspace(0, ys, 50); % Evaluation points across half-thickness

for k = 1:length(kappa_list)
    kappa = kappa_list(k);
    kappa_res = kappares_list(k);
    
    strain_profile = kappa .* y_list;
    
    % Lookup the actual stress experienced at those strains
    actual_stress = interp1(e_list, s_list, strain_profile, 'linear', 'extrap');
    
    % Calculate purely elastic stress "recovered" during unloading
    unloading_stress = E * (kappa - kappa_res) .* y_list;
    
    % Residual stress
    res_stress = actual_stress - unloading_stress;
    
    % Torque MUST be zero for equilibrium
    loss_torque = trapz(y_list, res_stress .* y_list);
    
    total_loss_torque = total_loss_torque + abs(loss_torque);
end

loss = total_loss_torque;


end