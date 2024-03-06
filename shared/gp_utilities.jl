using PosteriorBounds
using GaussianProcesses
using LinearAlgebra
using Distributions
using DataSandbox

# Functions that facilitate GP regression in the general and uniform-error settings. Needs a (permanent) home.

# regress GPs in vanilla way
function regress_gps(dataset, distribution::Normal; rkhs_flag=false, max_T=size(dataset[1], 2), delta_flag=true)

    indims = size(dataset[1], 1)    # get the input and output dimensions
    outdims = size(dataset[2], 1)
    @assert indims >= outdims

    # train the GPs
    mean = MeanZero()
    # sigma here should always be 1.0
     # todo: tune and generalize the kernel length-scale
    kernel = SE(log2(0.4), log2(1.0))  
    gps = []
    if rkhs_flag
        # todo: another parameter that needs to be tuned and generalized 
        lognoise = log10(0.01 + 1/max_T)
    else
        lognoise = log10(distribution.σ)
    end
    for i in 1:outdims
        if delta_flag
            gp = GP(dataset[1], dataset[2][i,:] - dataset[1][i,:], mean, kernel, lognoise)
        else
            gp = GP(dataset[1], dataset[2][i,:], mean, kernel, lognoise)
        end
        push!(gps, gp)
    end
    return gps
end

function regress_and_build(dataset, normal_distribution::Normal; rkhs_flag=false, max_T=size(dataset[1], 2), delta_flag=false)
    gps = regress_gps(dataset, normal_distribution, rkhs_flag=rkhs_flag, max_T=max_T, delta_flag=delta_flag)
    image_fcn, sigma_fcn = build_gp_image_fcn(gps, delta_flag=delta_flag)
    return gps, image_fcn, sigma_fcn
end

# build image function
function build_gp_image_fcn(gps; delta_flag=false, sample_flag=false)
    # calculate the theta vectors
    theta_vecs = []
    theta_vecs_train = []
    posterior_gps = []
    for i in eachindex(gps)
        kernel = PosteriorBounds.SEKernel(gps[i].kernel.σ2, gps[i].kernel.ℓ2) 
        theta_vec, theta_vec_train_squared = PosteriorBounds.theta_vectors(gps[i].x, kernel)
        push!(theta_vecs, theta_vec)
        push!(theta_vecs_train, theta_vec_train_squared)
        gp_ex = PosteriorBounds.PosteriorGP(
            gps[i].dim,
            gps[i].nobs,
            gps[i].x,
            gps[i].cK,
            Matrix{Float64}(undef, gps[i].nobs, gps[i].nobs),
            UpperTriangular(zeros(gps[i].nobs, gps[i].nobs)),
            inv(gps[i].cK),
            gps[i].alpha,
            kernel
        ) 
        PosteriorBounds.compute_factors!(gp_ex)
        push!(posterior_gps, gp_ex)
    end

    nthreads = Threads.nthreads()
    preallocs = [PosteriorBounds.preallocate_matrices(posterior_gps[1].dim, posterior_gps[1].nobs) for _ in 1:nthreads]

    # todo: fix up these function builders
    # if sample_flag
    # image_fcn(lower, upper; thread_idx=1) = get_image(lower, upper, posterior_gps, theta_vecs_train, theta_vecs, preallocs, 3000; thread_idx=thread_idx) 
    # else
    image_fcn(lower, upper; thread_idx=1) = get_image(lower, upper, posterior_gps, theta_vecs_train, theta_vecs, preallocs, delta_flag; thread_idx=thread_idx)
    # end

    # sigma_fcn(lower, upper; thread_idx=1) = get_σ_ub(lower, upper, posterior_gps, theta_vecs_train, theta_vecs, preallocs; thread_idx=thread_idx)

    sigma_fcn(lower, upper; thread_idx=1) = get_σ_ub_sample(lower, upper, posterior_gps, theta_vecs_train, theta_vecs, preallocs, 3000; thread_idx=thread_idx)

    return image_fcn, sigma_fcn
end

function get_image(lower, upper, posterior_gps, theta_vecs_train, theta_vecs, preallocs, delta_flag::Bool; thread_idx=1)
    res = zeros(length(posterior_gps), 2)
    for i in eachindex(gps)
        _, min_LB, _ = PosteriorBounds.compute_μ_bounds_bnb(posterior_gps[i], lower, upper, theta_vecs_train[i], theta_vecs[i], prealloc=preallocs[thread_idx])
        _, _, max_UB = PosteriorBounds.compute_μ_bounds_bnb(posterior_gps[i], lower, upper, theta_vecs_train[i], theta_vecs[i], max_flag=true, prealloc=preallocs[thread_idx])
        @assert min_LB <= max_UB

        if delta_flag
            res[i,1] = lower[i] + min_LB 
            res[i,2] = upper[i] + max_UB
        else
            res[i,1] = min_LB
            res[i,2] = max_UB
        end
    end
    return res[:,1], res[:,2]
end

function uniform_sample(lower::Vector{Float64}, upper::Vector{Float64}, N_samples)
    return vcat([hcat(rand(Uniform(lower[i], upper[i]), N_samples)...) for i in 1:length(lower)]...)
end

function get_image(lower, upper, posterior_gps, theta_vecs_train, theta_vecs, preallocs, N_samples::Int; thread_idx=1)
    res = zeros(length(posterior_gps), 2)

    μ_h = zeros(1,1)
    K_h = zeros(posterior_gps[1].nobs, 1)
    x_samples = uniform_sample(lower, upper, N_samples)

    for i in eachindex(posterior_gps)
        min_LB = Inf
        max_UB = -Inf

        for s in eachcol(x_samples) 
            # evaluate the mean functon
            PosteriorBounds.compute_μ!(μ_h, K_h, posterior_gps[i], s);
            val = μ_h[1]

            if val < min_LB
                min_LB = val
            end
            if val > max_UB
                max_UB = val
            end
        end

        @assert min_LB <= max_UB
        res[i,1] = lower[i] + min_LB
        res[i,2] = upper[i] + max_UB
    end
    return res[:,1], res[:,2]
end

function get_σ_ub(lower, upper, posterior_gps, theta_vecs_train, theta_vecs, preallocs; thread_idx=1)
    res = zeros(length(posterior_gps))
    for i in eachindex(gps)
        _, _, σ_ub = PosteriorBounds.compute_σ_bounds(posterior_gps[i], lower, upper, theta_vecs_train[i], theta_vecs[i], inv(posterior_gps[i].cK), prealloc=preallocs[thread_idx], max_iterations=100, bound_epsilon=1e-6,)
        res[i] = σ_ub
    end
    return res
end

function get_σ_ub_sample(lower, upper, posterior_gps, theta_vecs_train, theta_vecs, preallocs, N_samples; thread_idx=1)
    res = zeros(length(posterior_gps), 2)

    σ_h = zeros(1,1)
    x_samples = uniform_sample(lower, upper, N_samples)

    max_UB = -Inf
    for s in eachcol(x_samples) 
        # evaluate the sigma functon
        PosteriorBounds.compute_σ2!(σ_h, posterior_gps[1], s);
        val = σ_h[1]

        if val > max_UB
            max_UB = val
        end
    end
    return max_UB
end

# RKHS terms
mutable struct UniformRKHSError <: UniformError 
    sigma::Float64
    info_bound::Float64
    f_sup::Float64
    kernel_length::Float64
    norm_bound::Float64
    log_noise::Float64
end

# extension of CDF for RKHS error
function cdf(d::UniformRKHSError, x::Real)
    if x < 0.0
        return 0.0
    end
    R = exp(d.log_noise)
    frac = x/(d.sigma)
    dbound = exp(-0.5*(1/R*(frac - d.norm_bound))^2 + d.info_bound + 1.)
    return 1. - min(dbound, 1.) # is this min needed?
end

function info_gain_bound(gp)
    σ_v2 = (1 + 2/(gp.nobs))
    return 0.5*gp.nobs*log(1 + 1/σ_v2)
end

function RKHS_norm_bound(gp, f_sup, state_radius)
    return f_sup / sqrt(exp(-1/2*(2*state_radius)^2/exp(sqrt(gp.kernel.ℓ2))))
end

# todo: implementation of another uniform error type
# struct UniformPracticalError <: UniformError
#     sigma::Float64
#     log_det_info::Float64
#     f_sup::Float64
#     kernel_length::Float64
#     norm_bound::Float64
#     log_noise::Float64
# end

# function cdf(d::UniformPracticalError, x::Real)
#     if x < 0.0
#         return 0.0
#     end
#     R = exp(d.log_noise)
#     frac = x/(d.sigma)
#     dbound = exp(-0.5*(1/R*(frac - d.norm_bound))^2 + d.log_det_info)
#     return 1. - min(dbound, 1.) # is this min needed?
# end

