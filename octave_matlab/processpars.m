function [pars] = processpars(pars)
%PROCESSPARS sets PMB parameters to some default values

if nargin < 1
    pars.M = 5;
    pars.gtol = 1.0e-5;
    pars.ftol = 1.0e-8;
    pars.display = false;
    pars.message = true;
    pars.history = false;
    pars.maxiter = 1000;
    pars.maxiniter = 100;
    pars.maxfcalls = 1000;
    pars.maxtime = 3600;
else
    if ~isfield(pars, 'M')
        pars.M = 5;
    end
    
    if ~isfield(pars, 'gtol')
        pars.gtol = 1.0e-5;
    end
    
    if ~isfield(pars, 'ftol')
        pars.ftol = 1.0e-8;
    end
    
    if ~isfield(pars, 'display')
        pars.display = false;
    end
    
    if ~isfield(pars, 'message')
        pars.message = true;
    end
    
    if ~isfield(pars, 'history')
        pars.history = false;
    end
    
    if ~isfield(pars, 'maxiter')
        pars.maxiter = 1000;
    end
    
    if ~isfield(pars, 'maxiniter')
        pars.maxiniter = 100;
    end
    
    if ~isfield(pars, 'maxfcalls')
        pars.maxfcalls = 1000;
    end
    
    if ~isfield(pars, 'time')
        pars.maxtime = 3600;
    end
end

end % function ends