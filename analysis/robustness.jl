using CSV
using DataFrames
using Statistics
using Plots
using HypothesisTests
using GLM

# Function to load data
function load_data(file_path::String)::DataFrame
    df = CSV.read(file_path, DataFrame)
    
    # Convert string columns with numerical values in brackets to actual numbers
    function convert_column!(df::DataFrame, col_name::String)
        df[!, col_name] = [parse.(Float64, filter(x -> !isempty(x), split(replace(strip(x), r"[\[\]]" => ""), r"\s+"))) for x in df[!, col_name]]
    end

    columns_to_convert = [
        "Original Age Prediction", "Adversarial Age Prediction", 
        "Original Gender Prediction", "Adversarial Gender Prediction"
    ]
    
    for col in columns_to_convert
        convert_column!(df, col)
    end
    
    return df
end

# Function to get basic statistics for all prediction types
function get_basic_statistics(data::DataFrame)
    return describe(data)
end

# Function to calculate correlations between original and adversarial predictions
function calculate_correlations(data::DataFrame)::Dict{String, Float64}
    correlations = Dict(
        "Adversarial Age" => cor(reduce(vcat, data[!, "Original Age Prediction"]), reduce(vcat, data[!, "Adversarial Age Prediction"])),
        "Adversarial Gender" => cor(reduce(vcat, data[!, "Original Gender Prediction"]), reduce(vcat, data[!, "Adversarial Gender Prediction"]))
    )
    return correlations
end

# Function to perform paired t-tests for age and gender predictions
function perform_t_tests(data::DataFrame)::Dict{String, Any}
    t_tests = Dict(
        "Adversarial Age" => HypothesisTests.OneSampleTTest(reduce(vcat, data[!, "Original Age Prediction"]) .- reduce(vcat, data[!, "Adversarial Age Prediction"])),
        "Adversarial Gender" => HypothesisTests.OneSampleTTest(reduce(vcat, data[!, "Original Gender Prediction"]) .- reduce(vcat, data[!, "Adversarial Gender Prediction"]))
    )
    return t_tests
end

# Function to perform regression analysis for each type of perturbation
function perform_regression_analysis(data::DataFrame)::Dict{String, StatsModels.TableRegressionModel}
    regressions = Dict(
        "Adversarial Age" => lm(@formula(Adversarial_Age ~ Original_Age), DataFrame(Adversarial_Age = reduce(vcat, data[!, "Adversarial Age Prediction"]), Original_Age = reduce(vcat, data[!, "Original Age Prediction"]))),
        "Adversarial Gender" => lm(@formula(Adversarial_Gender ~ Original_Gender), DataFrame(Adversarial_Gender = reduce(vcat, data[!, "Adversarial Gender Prediction"]), Original_Gender = reduce(vcat, data[!, "Original Gender Prediction"])))
    )
    return regressions
end

# Function to plot scatter plots between original and adversarial predictions
function plot_scatter_plots(data::DataFrame)
    plot(reduce(vcat, data[!, "Original Age Prediction"]), reduce(vcat, data[!, "Adversarial Age Prediction"]),
         xlabel="Originalna procjena dobi", ylabel="Adversarialna procjena dobi",
         title="Originalna vs Adversarialna procjena dobi", seriestype=:scatter)
    savefig("robustness/original_vs_adversarial_age.png")

    plot(reduce(vcat, data[!, "Original Gender Prediction"]), reduce(vcat, data[!, "Adversarial Gender Prediction"]),
         xlabel="Originalna procjena spola", ylabel="Adversarialna procjena spola",
         title="Originalna vs Adversarialna procjena spola", seriestype=:scatter)
    savefig("robustness/original_vs_adversarial_gender.png")
end

# Function to plot Bland-Altman plots for age and gender predictions
function plot_bland_altman(data::DataFrame)
    function bland_altman_plot(original, perturbed, title, file_name)
        mean_vals = (original .+ perturbed) ./ 2
        diff_vals = original .- perturbed
        mean_diff = mean(diff_vals)
        std_diff = std(diff_vals)
        
        plt = scatter(mean_vals, diff_vals,
                      title=title,
                      xlabel="Srednja vrijednost originalne i adversarialne", ylabel="Razlika",
                      legend=false)
        hline!(plt, [mean_diff, mean_diff + 1.96 * std_diff, mean_diff - 1.96 * std_diff],
               label=["Prosječna razlika" "Prosječna + 1.96 SD" "Prosječna - 1.96 SD"], color=[:red :green :green])
        savefig(plt, file_name)
    end

    bland_altman_plot(reduce(vcat, data[!, "Original Age Prediction"]), reduce(vcat, data[!, "Adversarial Age Prediction"]), "Bland-Altman Graf: Originalna vs Adversarialna procjena dobi", "robustness/bland_altman_adversarial_age.png")
    bland_altman_plot(reduce(vcat, data[!, "Original Gender Prediction"]), reduce(vcat, data[!, "Adversarial Gender Prediction"]), "Bland-Altman Graf: Originalna vs Adversarialna procjena spola", "robustness/bland_altman_adversarial_gender.png")
end

# Function to save statistical summary
function save_statistical_summary(data::DataFrame, correlations::Dict{String, Float64}, t_tests::Dict{String, Any}, regressions::Dict{String, StatsModels.TableRegressionModel})
    open("robustness/statistical_summary.txt", "w") do f
        println(f, "Osnovna statistika:\n", describe(data))
        
        println(f, "\nKorelacije između originalnih i adversarialnih predikcija:")
        for (k, v) in correlations
            println(f, "$k: $v")
        end

        println(f, "\nT-test rezultati za razlike između originalnih i adversarialnih predikcija:")
        for (k, v) in t_tests
            println(f, "$k: ", v)
        end

        println(f, "\nRezultati regresijske analize:")
        for (k, v) in regressions
            println(f, "$k: ", coeftable(v))
        end
    end
end

# Main function
function main()
    file_path = "../results/robustness.csv"
    data = load_data(file_path)
    
    println("Osnovna statistika:\n", get_basic_statistics(data))
    
    correlations = calculate_correlations(data)
    println("Korelacije:\n", correlations)
    
    t_tests = perform_t_tests(data)
    println("T-test rezultati:\n", t_tests)
    
    regressions = perform_regression_analysis(data)
    println("Rezultati regresijske analize:\n", regressions)
    
    plot_scatter_plots(data)
    plot_bland_altman(data)
    
    save_statistical_summary(data, correlations, t_tests, regressions)
end

# Execute main function
main()
