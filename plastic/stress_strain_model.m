% =========================================================================
% Script Name: stress_strain_model.m
% Description: Centralized hub for evaluating various constitutive models.
%              Maps strain to stress for elastoplastic and damage materials.
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

% =========================================================================
% FUNCTION: stress_strain_model
% =========================================================================
% Centralized hub for all constitutive models.
% Inputs:
%   e          : Strain array (can be scalar, vector, or matrix)
%   model_type : String defining the model to use
%   x          : Array of optimized parameters (scaled by CMA-ES)
%   E          : Young's Modulus (Pa)
% Outputs:
%   s          : True/Engineering Stress array (Pa)
function s = stress_strain_model(e, model_type, x, E)
s = zeros(size(e));

if strcmpi(model_type, 'Voce')
    % x = [ey, smax_factor, m_factor]
    ey = x(1); 
    smax = x(2) * E * 0.01; 
    m = x(3) * 100;
    
    idx_el = e <= ey; 
    idx_pl = ~idx_el;
    
    s(idx_el) = E * e(idx_el);
    s(idx_pl) = E * ey + (smax - E * ey) * (1 - exp(-m * (e(idx_pl) - ey)));
    
elseif strcmpi(model_type, 'GSell_Jonas_Hybrid')
    % x = [ey, K_factor, h_factor]
    ey = x(1);
    K = x(2) * E * 0.01; 
    h = x(3) * 10;
    
    % Mathematically lock 'w' to enforce perfect slope continuity (C1)
    w = E / K;
    
    idx_el = e <= ey; 
    idx_pl = ~idx_el;
    
    s(idx_el) = E * e(idx_el);
    
    % Shift plastic strain so the curve connects continuously at the yield point
    e_pl = e(idx_pl) - ey;
    s(idx_pl) = E * ey + K .* (1 - exp(-w .* e_pl)) .* exp(h .* e_pl.^2);
    
elseif strcmpi(model_type, 'Orthogonal_Piecewise')
    % x = [ey, sp_factor, beta_factor]
    ey = x(1); 
    sp = x(2) * E * 0.01; 
    beta = x(3) * 100;
    sy = E * ey;
    
    idx_el = e <= ey; 
    idx_pl = ~idx_el;
    
    s(idx_el) = E * e(idx_el);
    s(idx_pl) = sp + (sy - sp) * exp(-beta * (e(idx_pl) - ey));
    
elseif strcmpi(model_type, 'CDM')
    % x = [ey, c_factor]
    ey = x(1); 
    c = x(2) * 100;
    
    idx_el = e <= ey; 
    idx_pl = ~idx_el;
    
    s(idx_el) = E * e(idx_el);
    % D = 1 - exp(-c*(e - ey)) -> Stress = E * e * (1 - D)
    s(idx_pl) = (E * e(idx_pl)) .* exp(-c * (e(idx_pl) - ey));
    
else
    error('Unknown constitutive model: %s', model_type);
end


end