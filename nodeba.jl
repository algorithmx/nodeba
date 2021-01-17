#: nodeba.jl
#: a simple program to run julia functions on pre-allocated nodes in parallel

using Distributed

global const _nodeba_banner = [
    "",
    " * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * ",
    "",
    "     _       _      ______      _______                    _______         ____",
    "    |+\\     |+|    /+/  \\+\\    |+|   \\+\\    |+|=======    |+|===\\+\\       /+/\\+\\",
    "    |++\\    |+|   |+|    |+|   |+|    \\+\\   |+|           |+|    |+|     /+/  \\+\\",
    "    |+ +\\   |+|   |+|    |+|   |+|    |+|   |+|=======    |+|====/      |+|    |+|",
    "    |+|\\+\\  |+|   |+|    |+|   |+|    |+|   |+|           |+|    |+|    |+|====|+|",
    "    |+| \\+\\ |+|   |+|    |+|   |+|    /+/   |+|           |+|    /+/    |+|    |+|",
    "    |+|  \\+\\|+|    \\+\\__/+/    |+|___/+/    |+|=======    |+|===/+/     |+|    |+|",
    "",
    "     . . . . . . . . . . . . . NODEBA : KEEP YOUR NODES BUSY . . . . . . . . . . . . .",
    "     . . . . . . . . . . . . . AUTHOR : YUNLONG LIAN . . . . . . . . . . . . . . . . .",
    "",
    "    nodeba.jl  Version 0.1",
    "",
    " * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * ",
]

##

# HPC typically has a name convention for nodes. 
# Here I assumes SLURM_JOB_NODELIST has the following format 
# p[12-14],o[23]
# which means that we have p12 , p13 , p14 and o23
# sometimes the full domain name has to be specified
# with node name as prefix
# node_name_translator(x) transforms string x into the correct form 
# to be used by addprocs
function parse_node_list(nl, node_name_translator::Function)::Vector{String}
    rstr = r"[A-Za-z]+\[((\d+\-)?\d+,)*((\d+\-)?\d+)\]"
    substr_pos = findall(rstr, nl)
    if length(substr_pos)==0  return []  end
    @inline to_range(a,b) = collect(a:b)
    @inline p_int(s)      = parse(Int,s)
    @inline prs(s)        = (occursin("-",s) ? to_range((p_int.(split(s,"-",keepempty=false)))...) : [p_int(s),])
    ret = []
    for p ∈ substr_pos
        head,n1 = split(nl[p],"[",keepempty=false)
        n2 = rstrip(n1, ']')
        n3 = split(n2,",",keepempty=false)
        n4 = vcat([prs(i) for i in n3]...)
        for j ∈ n4
            push!(ret, "$(head)$(j)")
        end
    end
    return node_name_translator.(ret)
end


# keys important to parallel computing
#TODO other evn. varibles ?
function get_slurm_config(env, f::Function)
    return  Dict(   :NODELIST     => parse_node_list(get(env,"SLURM_JOB_NODELIST",""), f),
                    :NUM_NODES    => parse(Int,get(env,"SLURM_JOB_NUM_NODES","0")),
                    :CPUS_ON_NODE => parse(Int,get(env,"SLURM_CPUS_ON_NODE","1"))   )
end


function check_distributed_env(
    P::Vector{Int}  # a collection of worker ids.
    )

    # the master worker always have id=1
    # the program starts at this worker
    master    = remotecall_fetch(x->"Master node : id=$(myid()), hostname=$(gethostname())", 1, 1)
    
    # other nodes are added via addprocs later
    wok_info  = pmap(   x->"$x : id=$(myid()), hostname=$(gethostname())", WorkerPool(P), P, 
                        retry_delays=ExponentialBackOff(n=5)    )
    
    NWOKS = length(P)
    
    info = [
        "",
        "--------------- NODE  INFO ----------------",
        "",
        "Number of workers : $NWOKS",
        master,
        wok_info...,
        "",
        "-------------------------------------------",
        "",
    ]
    return join(info, "\n")
end


#: debug mode
#+ delete printlns
function launch_nodeba(;
    mode=:single_process_per_node,
    topology=:all_to_all,
    quiet=false, 
    node_name_translator=x->x
    )

    @assert mode ∈ [:single_process_per_node, :maximum_process_per_node]

    if !quiet  @info join(_nodeba_banner, "\n")  end
    slurm_conf = get_slurm_config(ENV, node_name_translator)

    if !( length(slurm_conf[:NODELIST])>0 && slurm_conf[:NUM_NODES]>0 )
        @error "launch_nodeba() failed. \nslurm_conf[:NODELIST]=$(slurm_conf[:NODELIST])\nslurm_conf[:NUM_NODES]=$(slurm_conf[:NUM_NODES])"
    end

    NODELIST = []
    if mode == :single_process_per_node
        println("addprocs starts at : $(Int(time_ns()))")
        flush(stdout) ; flush(stdout) ;
        NODELIST = addprocs(slurm_conf[:NODELIST], topology=topology)
        println("addprocs finishes at : $(Int(time_ns()))")
        flush(stdout) ; flush(stdout) ;
    elseif mode == :maximum_process_per_node
        NL   = slurm_conf[:NODELIST]
        NCPU = zeros(Int,length(NL)) .+ slurm_conf[:CPUS_ON_NODE]
        MACH = collect(zip(NL,NCPU))

        println("addprocs starts at : $(Int(time_ns()))")
        flush(stdout) ; flush(stdout) ;
        NODELIST = addprocs(MACH, topology=topology)
        println("addprocs finishes at : $(Int(time_ns()))")
        flush(stdout) ; flush(stdout) ;
    end

    println("check_distributed_env starts at : $(Int(time_ns()))")
    flush(stdout) ; flush(stdout) ;
    @info check_distributed_env(NODELIST)
    println("check_distributed_env ends at : $(Int(time_ns()))")
    flush(stdout) ; flush(stdout) ;

    return slurm_conf, NODELIST
end

##* ========== TEST ==========

timenow(i=0) = println("***** Time now $i **  $(Int(time_ns()))")

timenow(0); flush(stdout);

slurm_conf, NODELIST = launch_nodeba(
                            mode=:maximum_process_per_node,
                            topology=:master_worker )

timenow(1); flush(stdout);

using Statistics

timenow(3); flush(stdout);

@everywhere function findpi(n)
    inside = 0
    for i = 1:n
        x = rand(); y = rand()
        inside += (x^2 + y^2) <= 1
    end
    return 4 * inside / n
end

pfindpi(N) = mean( pmap(n->findpi(n), Int64[N ÷ nworkers() for i=1:nworkers()]) )

timenow(4); flush(stdout);

println("pfindpi(192_000_000_000) costs :")
@time k = pfindpi(6400_000_000_000)
println("k = $k")


timenow(5); flush(stdout);

println("pfindpi(192_000_000_000) costs :")
@time k = pfindpi(6400_000_000_000)
println("k = $k")

timenow(6); flush(stdout);

t = rmprocs(NODELIST...; waitfor=10)

wait(t)

timenow(7); flush(stdout);

exit()
