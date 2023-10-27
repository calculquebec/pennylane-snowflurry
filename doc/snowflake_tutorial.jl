using Snowflurry

function groverOracle(c::QuantumCircuit, cibles::Vector{String})
    qubits=c.qubit_count
    for cible in cibles    
        for i in range(1,c.qubit_count)
            if cible[i] == '0'
                push!(c,sigma_x(i))
            end
        end
        mcz=controlled(sigma_z(qubits),collect(1:qubits-1))
        push!(c,mcz)
        for i in range(1,length(cible))
            if cible[i] == '0'
                push!(c,sigma_x(i))
            end
        end
    end
end;


function groverOperateur(c::QuantumCircuit)
    qubits=c.qubit_count
    for i in range(1,qubits)
        push!(c,hadamard(i),sigma_x(i))
    end

    mcz=controlled(sigma_z(qubits),collect(1:qubits-1))
    push!(c,mcz)
    for i in range(1,qubits)
        push!(c,sigma_x(i),hadamard(i))
    end      
end;


function circuitGrover(qubits::Int,cibles)
    c=QuantumCircuit(qubit_count=qubits)

    for i in range(1,c.qubit_count)
        push!(c,hadamard(i))
    end

    iterations=Int(floor(pi/4 * sqrt(length(cibles)/(2^qubits))))
    for i in range(0,iterations)
        groverOracle(c,cibles)
        groverOperateur(c)
    end
    return c
end;

#éléments cibles marqués
cibles=["0000","1111","0110"]
qubits=length(cibles[1])

circuit=circuitGrover(qubits,cibles)

print(circuit)
print(simulate(circuit))
    
using SnowflurryPlots
plot_histogram(circuit, 200)


# max(simulate(circuit))
resultats=Dict()
maximum=0
maxVal="0000"
for i in range(0,15)
    resultats[string(i,base=2,pad=4)]=0
end
for resultat in simulate_shots(circuit, 100)
    resultats[resultat]+=1
    if resultats[resultat]>maximum
        maximum=resultats[resultat]
        maxVal=resultat
    end
end
# print(resultats)
print(maxVal)