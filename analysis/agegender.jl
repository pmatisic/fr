using CSV
using DataFrames
using Statistics
using Plots
using StatsPlots
using HypothesisTests

# Function to load data
function load_data(file_path::String)::DataFrame
    data = CSV.read(file_path, DataFrame)
    return data
end

# Function to get basic statistics
function get_basic_statistics(data::DataFrame)
    return describe(select(data, [:Age, :Gender]))
end

# Function to save plots
function save_plot(plot, filename::String)
    savefig(plot, filename)
end

# Function to check for missing values
function check_missing_values(data::DataFrame)
    return [sum(ismissing(col)) for col in eachcol(data)]
end

# Function to plot age distribution
function plot_age_distribution(data::DataFrame)
    plt = histogram(data.Age, bins=30, 
                    title="Distribucija godina", 
                    xlabel="Godine", ylabel="Frekvencija",
                    label="Godine",
                    legend=false)
    save_plot(plt, "agegender/age_distribution.png")
end

# Function to plot gender distribution
function plot_gender_distribution(data::DataFrame)
    gender_counts = combine(groupby(data, :Gender), nrow => :Count)
    
    colors = ["#1f77b4", "#ff7f0e"]
    
    plt = pie(gender_counts.Count, title="Distribucija spola", color=colors, legend=false)
    
    plot!(legend=:outertopright)
    scatter!([NaN], [NaN], m=(12, :circle, 0.8, colors[1]), label="Muški")
    scatter!([NaN], [NaN], m=(12, :circle, 0.8, colors[2]), label="Ženski")
    
    savefig(plt, "agegender/gender_distribution.png")
end

# Function to plot age by gender
function plot_age_by_gender(data::DataFrame)
    plt = boxplot(data.Gender, data.Age, 
                  title="Godine po spolu", 
                  xlabel="Spol", ylabel="Godine",
                  xticks=(1:2, ["Muški", "Ženski"]))
    save_plot(plt, "agegender/age_by_gender.png")
end

# Function to perform t-test
function perform_t_test(data::DataFrame)
    male_ages = data[data.Gender .== "Male", :Age]
    female_ages = data[data.Gender .== "Female", :Age]
    return HypothesisTests.EqualVarianceTTest(male_ages, female_ages)
end

# Function to save statistical summary
function save_statistical_summary(data::DataFrame, t_test_result)
    open("agegender/statistical_summary.txt", "w") do f
        println(f, "Osnovna statistika:\n", describe(select(data, [:Age, :Gender])))
        println(f, "\nProvjera nedostajućih vrijednosti:\n", check_missing_values(data))
        println(f, "\nT-test rezultat:\n", t_test_result)
    end
end

# Main function
function main()
    file_path = "../results/agegender.csv"
    data = load_data(file_path)
    
    println("Osnovna statistika:\n", get_basic_statistics(data))
    
    println("Provjera nedostajućih vrijednosti:\n", check_missing_values(data))
    
    plot_age_distribution(data)
    plot_gender_distribution(data)
    plot_age_by_gender(data)
    
    t_test_result = perform_t_test(data)
    println("T-test rezultat:\n", t_test_result)
    
    save_statistical_summary(data, t_test_result)
end

# Execute main function
main()
