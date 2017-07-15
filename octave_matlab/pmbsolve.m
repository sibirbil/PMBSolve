function [output] = pmbsolve(fun, x0, pars)
%PMBSOLVE is a solver for unconstrained optimization problems of the form:
%
%                           minimize f(x)
%
%   where f is a real-valued differentiable function that works on
%   n-dimensional vectors.
%
%   Usage 1:
%   OUTPUT = pmbsolve(FUN, X0)
%     FUN is the handle to the real-valued function. This function should
%     return the function value and the gradient vector at a given point.
%     X0 is the initial starting solution.
%
%   Usage 2:
%   OUTPUT = pmbsolve(FUN, X0, PARS)
%     The last argument PARS is optional. It specifies different options
%     for the optimization procedure. Here is a list of the options with
%     their default values given in parantheses:
%      PARS.M: The memory size for the preconditioner (5)
%      PARS.gtol: The tolerance for the first-order optimality (1.0e-5)
%      PARS.ftol: The normalized difference between two consecutive function values (1.0e-8)
%      PARS.display: To display information about the progress in every iteration (false)
%      PARS.message: Shows a message about the exit condition or errors (true)
%      PARS.history: Stores the function values and the first-order errors throughout the iterations (false)
%      PARS.maxiter: Maximum number of iterations (1000)
%      PARS.maxiniter: Maximum number of inner iterations for model building (100)
%      PARS.maxfcalls: Maximum number of function calls (1000)
%      PARS.maxtime: Maximum computation time in seconds (3600)
%
%     The solver returns OUTPUT consisting of the following fields:
%      OUTPUT.fcalls: Total number of function calls to solve the problem
%      OUTPUT.nmbs: Total number of times a model is built during the inner iterations
%      OUTPUT.exit: Exit status
%        1: First order condition is met (gtol)
%        0: Maximum number of inner iterations is reached (maxiniter)
%       -1: Maximum number of iterations is reached (maxiter)
%       -2: Maximum number of function calls is reached (maxfcalls)
%       -3: Time limit is reached (maxtime)
%       -4: Change in function value between two consecutive iterations is below tolerance (ftol)
%      OUTPUT.time: Time to solve the problem in seconds
%      OUTPUT.x: Final solution
%      OUTPUT.g: Final gradient at OUTPUT.x
%      OUTPUT.niter: Number of iterations

if nargin < 3
    pars = processpars();
else
    pars = processpars(pars);
end

output.fcalls = 0;
output.nmbs = 0;
if (pars.history)
    output.fhist = [];
    output.nghist = [];
end

tstart = tic;
x = x0;
[f,g] = fun(x);
output.fcalls = output.fcalls+1;
fold = Inf;
n = length(x);
S = zeros(n, pars.M);
Y = zeros(n, pars.M);
YS = zeros(pars.M, 1);
mem_start = 1;
mem_end = 0;
Hdiag = 1.0;

iter = 0;

while(true) % Outer Iterations ---->>

    %% Stopping Conditions ---->>
    ngf = norm(g,'inf');
    if(ngf < pars.gtol)
        output.exit = 1;
        if (pars.message)
            fprintf('First order condition is within gtol\n');
        end
        break;
    end
    if (iter >= pars.maxiter)
        output.exit = -1;
        if (pars.message)
            fprintf('Maximum number of iterations (maxiter) is reached\n');
        end
        break;
    end
    if (output.fcalls >= pars.maxfcalls)
        output.exit = -2;
        if (pars.message)
            fprintf('Maximum number of function calss (maxfcalls) is reached\n');
        end
        break;
    end
    if (toc(tstart) >= pars.maxtime)
        output.exit = -3;
        if (pars.message)
            fprintf('Maximum time limit (maxtime) is reached\n');
        end
        break;
    end
    if ((fold - f)/max([abs(fold), abs(f), 1]) < pars.ftol)
        output.exit = -4;
        if (pars.message)
            fprintf('Function value decreases less than ftol\n');
        end
        break;
    else
        fold = f;
    end
    %% <<---- Stopping Conditions

    if (pars.history)
        output.fhist(iter) = f;
        output.nghist(iter) = ngf;
    end

    if (pars.display)
        fprintf('Iter: %d ===> f = %f \t norm(g) = %f\n', iter, f, ngf);
    end

    %% L-BFGS Preconditioning ---->>
    if (iter >= 1)
        [s, Hdiag, S, Y, YS, mem_start, mem_end] = ...
	  precond(g, g_old, s, Hdiag, S, Y, YS, pars.M, mem_start, mem_end);
    else
        s = -g/ngf; % Normalized steepest descent direction
    end
    g_old = g;
    %% <<---- L-BFGS Preconditioning

    initer = 0;
    while(initer < pars.maxiniter) % Inner Iterations ---->>

        xt = x+s; % (x^k_t)

        [ft,gt] = fun(xt); % (f^k_t) and (g^k_t)
        output.fcalls = output.fcalls+1;

        sg = s'*g; % (v6)

        if(f - ft > -1.0e-4*sg)
            x = xt;
            f = ft;
            g = gt;
            break;
        end

        sgt = s'*gt;

        y = gt-g;  % (y^k_t)
        ys = sgt-sg; % (v1) = y'*s
        ss = s'*s; % (v2)
        yy = y'*y; % (v3)
        yg = y'*g; % (v4)
        gg = g'*g; % (v5)

        %% Guess eta  ---->>
        fdiff = abs(f - ft);
        if (abs(sg) > 1.0e-8)
          eta1 = fdiff/abs(sg);
        else
          eta1 = 1.0;
        end
        if (abs(sgt) > 1.0e-8)
          eta2 = fdiff/abs(sgt);
        else
          eta2 = 1.0;
        end
        eta = min(eta1, eta2)/(eta1 + eta2);
        %% <<---- Guess eta

        sigma = 1.0/2.0*(sqrt(ss)*(sqrt(yy)+1.0/eta*sqrt(gg))-ys);
        theta = (ys+2.0*sigma)^2.0-ss*yy;

        cg= -ss/(2.0*sigma); % (cg(sigma))
        cs = cg/theta*(-(ys+2.0*sigma)*yg+yy*sg); % (cs(sigma))
        cy = cg/theta*(-(ys+2.0*sigma)*sg+ss*yg); % (cy(sigma))

        s = cg*g+cs*s+cy*y; % Step

        initer = initer+1;

    end % <<---- Inner Iterations

    output.nmbs = output.nmbs+initer;

    if(initer >= pars.maxiniter)
        output.exit = 0;
        if (pars.message)
            fprintf('Maximum number of inner iterations (maxiniter) is reached\n');
        end
        break;
    end

    iter = iter+1;

end % <<---- Outer Iterations

output.time = toc(tstart);
output.x = x;
output.fval = f;
output.g = g;
output.niter = iter;

end % function ends
