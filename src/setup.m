function setup()
% SETUP  Update Matlab search path.
%
% References:
%   [1] VLFEAT http://www.vlfeat.org/
%   [2] SPAMS toolbox (http://spams-devel.gforge.inria.fr/)
%   [3] LIBSVM https://www.csie.ntu.edu.tw/~cjlin/libsvm/

% mjp, april 2016

here = fileparts(mfilename('fullpath'));
appsDir = fullfile('/Users', 'pekalmj1', 'Apps');


%% Subdirectories

addpath(fullfile(here, 'pooling'));
addpath(fullfile(here, 'features'));
addpath(fullfile(here, 'classification'));
addpath(fullfile(here, 'data'));
addpath(fullfile(here, 'examples'));



%% Third party dependencies

% Misc. third-party scripts
% Assumes you are running this script from pwd.
if ~exist('tight_subplot')
    thirdPartyDir = fullfile(here, 'thirdparty');
    addpath(thirdPartyDir);
    fprintf('[%s]: using "%s" for other third-party codes\n', ...
            mfilename, thirdPartyDir);
end


% Codes for SIFT [1]
if ~exist('vl_dsift')
    vlroot = fullfile(appsDir, 'vlfeat-0.9.20');
    run(fullfile(vlroot, 'toolbox', 'vl_setup'));
    fprintf('[%s]: using VLFeat from "%s"\n', mfilename, vlroot);
end

% LIBSVM [3]
if ~exist('svmpredict')
    libsvmDir = fullfile(appsDir, 'libsvm-3.21', 'matlab');
    addpath(libsvmDir, '-begin');
    fprintf('[%s]: using LIBSVM dir "%s"\n', mfilename, libsvmDir);
end



% Support for sparse coding [2].
% Not all experiments require sparse coding, so this can
% be optional.
%
% This is a slightly modified version of what start_spams.m does;
% except it does not require we be in any particular directory.
if ~exist('test_release')
    spamsroot = fullfile(appsDir, 'spams-matlab');
    addpath(fullfile(spamsroot, 'test_release'));
    addpath(fullfile(spamsroot, 'src_release'));
    addpath(fullfile(spamsroot, 'build'));
    setenv('MKL_NUM_THREADS','1');
    setenv('MKL_SERIAL','YES');
    setenv('MKL_DYNAMIC','NO');
    fprintf('[%s]: using SPAMS from "%s"\n', mfilename, spamsroot);
end

