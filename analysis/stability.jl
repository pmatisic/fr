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
        "Original Age Prediction", "Noisy Age Prediction", "Rotation Age Prediction", 
        "Brightness Age Prediction", "Contrast Age Prediction", "Original Gender Prediction",
        "Noisy Gender Prediction", "Rotation Gender Prediction", "Brightness Gender Prediction",
        "Contrast Gender Prediction"
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

# Function to calculate correlations between original and perturbed predictions
function calculate_correlations(data::DataFrame)::Dict{String, Float64}
    correlations = Dict(
        "Noisy Age" => cor(reduce(vcat, data[!, "Original Age Prediction"]), reduce(vcat, data[!, "Noisy Age Prediction"])),
        "Rotation Age" => cor(reduce(vcat, data[!, "Original Age Prediction"]), reduce(vcat, data[!, "Rotation Age Prediction"])),
        "Brightness Age" => cor(reduce(vcat, data[!, "Original Age Prediction"]), reduce(vcat, data[!, "Brightness Age Prediction"])),
        "Contrast Age" => cor(reduce(vcat, data[!, "Original Age Prediction"]), reduce(vcat, data[!, "Contrast Age Prediction"])),
        "Noisy Gender" => cor(reduce(vcat, data[!, "Original Gender Prediction"]), reduce(vcat, data[!, "Noisy Gender Prediction"])),
        "Rotation Gender" => cor(reduce(vcat, data[!, "Original Gender Prediction"]), reduce(vcat, data[!, "Rotation Gender Prediction"])),
        "Brightness Gender" => cor(reduce(vcat, data[!, "Original Gender Prediction"]), reduce(vcat, data[!, "Brightness Gender Prediction"])),
        "Contrast Gender" => cor(reduce(vcat, data[!, "Original Gender Prediction"]), reduce(vcat, data[!, "Contrast Gender Prediction"]))
    )
    return correlations
end

# Function to perform paired t-tests for age and gender predictions
function perform_t_tests(data::DataFrame)::Dict{String, Any}
    t_tests = Dict(
        "Noisy Age" => HypothesisTests.OneSampleTTest(reduce(vcat, data[!, "Original Age Prediction"]) .- reduce(vcat, data[!, "Noisy Age Prediction"])),
        "Rotation Age" => HypothesisTests.OneSampleTTest(reduce(vcat, data[!, "Original Age Prediction"]) .- reduce(vcat, data[!, "Rotation Age Prediction"])),
        "Brightness Age" => HypothesisTests.OneSampleTTest(reduce(vcat, data[!, "Original Age Prediction"]) .- reduce(vcat, data[!, "Brightness Age Prediction"])),
        "Contrast Age" => HypothesisTests.OneSampleTTest(reduce(vcat, data[!, "Original Age Prediction"]) .- reduce(vcat, data[!, "Contrast Age Prediction"])),
        "Noisy Gender" => HypothesisTests.OneSampleTTest(reduce(vcat, data[!, "Original Gender Prediction"]) .- reduce(vcat, data[!, "Noisy Gender Prediction"])),
        "Rotation Gender" => HypothesisTests.OneSampleTTest(reduce(vcat, data[!, "Original Gender Prediction"]) .- reduce(vcat, data[!, "Rotation Gender Prediction"])),
        "Brightness Gender" => HypothesisTests.OneSampleTTest(reduce(vcat, data[!, "Original Gender Prediction"]) .- reduce(vcat, data[!, "Brightness Gender Prediction"])),
        "Contrast Gender" => HypothesisTests.OneSampleTTest(reduce(vcat, data[!, "Original Gender Prediction"]) .- reduce(vcat, data[!, "Contrast Gender Prediction"]))
    )
    return t_tests
end

# Function to perform regression analysis for each type of perturbation
function perform_regression_analysis(data::DataFrame)::Dict{String, StatsModels.TableRegressionModel}
    regressions = Dict(
        "Noisy Age" => lm(@formula(Noisy_Age ~ Original_Age), DataFrame(Noisy_Age = reduce(vcat, data[!, "Noisy Age Prediction"]), Original_Age = reduce(vcat, data[!, "Original Age Prediction"]))),
        "Rotation Age" => lm(@formula(Rotation_Age ~ Original_Age), DataFrame(Rotation_Age = reduce(vcat, data[!, "Rotation Age Prediction"]), Original_Age = reduce(vcat, data[!, "Original Age Prediction"]))),
        "Brightness Age" => lm(@formula(Brightness_Age ~ Original_Age), DataFrame(Brightness_Age = reduce(vcat, data[!, "Brightness Age Prediction"]), Original_Age = reduce(vcat, data[!, "Original Age Prediction"]))),
        "Contrast Age" => lm(@formula(Contrast_Age ~ Original_Age), DataFrame(Contrast_Age = reduce(vcat, data[!, "Contrast Age Prediction"]), Original_Age = reduce(vcat, data[!, "Original Age Prediction"]))),
        "Noisy Gender" => lm(@formula(Noisy_Gender ~ Original_Gender), DataFrame(Noisy_Gender = reduce(vcat, data[!, "Noisy Gender Prediction"]), Original_Gender = reduce(vcat, data[!, "Original Gender Prediction"]))),
        "Rotation Gender" => lm(@formula(Rotation_Gender ~ Original_Gender), DataFrame(Rotation_Gender = reduce(vcat, data[!, "Rotation Gender Prediction"]), Original_Gender = reduce(vcat, data[!, "Original Gender Prediction"]))),
        "Brightness Gender" => lm(@formula(Brightness_Gender ~ Original_Gender), DataFrame(Brightness_Gender = reduce(vcat, data[!, "Brightness Gender Prediction"]), Original_Gender = reduce(vcat, data[!, "Original Gender Prediction"]))),
        "Contrast Gender" => lm(@formula(Contrast_Gender ~ Original_Gender), DataFrame(Contrast_Gender = reduce(vcat, data[!, "Contrast Gender Prediction"]), Original_Gender = reduce(vcat, data[!, "Original Gender Prediction"])))
    )
    return regressions
end

# Function to plot scatter plots between original and perturbed predictions
function plot_scatter_plots(data::DataFrame)
    plot(reduce(vcat, data[!, "Original Age Prediction"]), reduce(vcat, data[!, "Noisy Age Prediction"]),
         xlabel="Originalna procjena dobi", ylabel="Procjena dobi s bukom",
         title="Originalna vs Procjena dobi s bukom", seriestype=:scatter)
    savefig("stability/original_vs_noisy_age.png")

    plot(reduce(vcat, data[!, "Original Age Prediction"]), reduce(vcat, data[!, "Rotation Age Prediction"]),
         xlabel="Originalna procjena dobi", ylabel="Procjena dobi s rotacijom",
         title="Originalna vs Procjena dobi s rotacijom", seriestype=:scatter)
    savefig("stability/original_vs_rotation_age.png")

    plot(reduce(vcat, data[!, "Original Age Prediction"]), reduce(vcat, data[!, "Brightness Age Prediction"]),
         xlabel="Originalna procjena dobi", ylabel="Procjena dobi s promjenom osvjetljenja",
         title="Originalna vs Procjena dobi s promjenom osvjetljenja", seriestype=:scatter)
    savefig("stability/original_vs_brightness_age.png")

    plot(reduce(vcat, data[!, "Original Age Prediction"]), reduce(vcat, data[!, "Contrast Age Prediction"]),
         xlabel="Originalna procjena dobi", ylabel="Procjena dobi s promjenom kontrasta",
         title="Originalna vs Procjena dobi s promjenom kontrasta", seriestype=:scatter)
    savefig("stability/original_vs_contrast_age.png")

    plot(reduce(vcat, data[!, "Original Gender Prediction"]), reduce(vcat, data[!, "Noisy Gender Prediction"]),
         xlabel="Originalna procjena spola", ylabel="Procjena spola s bukom",
         title="Originalna vs Procjena spola s bukom", seriestype=:scatter)
    savefig("stability/original_vs_noisy_gender.png")

    plot(reduce(vcat, data[!, "Original Gender Prediction"]), reduce(vcat, data[!, "Rotation Gender Prediction"]),
         xlabel="Originalna procjena spola", ylabel="Procjena spola s rotacijom",
         title="Originalna vs Procjena spola s rotacijom", seriestype=:scatter)
    savefig("stability/original_vs_rotation_gender.png")

    plot(reduce(vcat, data[!, "Original Gender Prediction"]), reduce(vcat, data[!, "Brightness Gender Prediction"]),
         xlabel="Originalna procjena spola", ylabel="Procjena spola s promjenom osvjetljenja",
         title="Originalna vs Procjena spola s promjenom osvjetljenja", seriestype=:scatter)
    savefig("stability/original_vs_brightness_gender.png")

    plot(reduce(vcat, data[!, "Original Gender Prediction"]), reduce(vcat, data[!, "Contrast Gender Prediction"]),
         xlabel="Originalna procjena spola", ylabel="Procjena spola s promjenom kontrasta",
         title="Originalna vs Procjena spola s promjenom kontrasta", seriestype=:scatter)
    savefig("stability/original_vs_contrast_gender.png")
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
                      xlabel="Srednja vrijednost originalne i perturbirane", ylabel="Razlika",
                      legend=false)
        hline!(plt, [mean_diff, mean_diff + 1.96 * std_diff, mean_diff - 1.96 * std_diff],
               label=["Prosječna razlika" "Prosječna + 1.96 SD" "Prosječna - 1.96 SD"], color=[:red :green :green])
        savefig(plt, file_name)
    end

    bland_altman_plot(reduce(vcat, data[!, "Original Age Prediction"]), reduce(vcat, data[!, "Noisy Age Prediction"]), "Bland-Altman Graf: Originalna vs Procjena dobi s bukom", "stability/bland_altman_noisy_age.png")
    bland_altman_plot(reduce(vcat, data[!, "Original Age Prediction"]), reduce(vcat, data[!, "Rotation Age Prediction"]), "Bland-Altman Graf: Originalna vs Procjena dobi s rotacijom", "stability/bland_altman_rotation_age.png")
    bland_altman_plot(reduce(vcat, data[!, "Original Age Prediction"]), reduce(vcat, data[!, "Brightness Age Prediction"]), "Bland-Altman Graf: Originalna vs Procjena dobi s promjenom osvjetljenja", "stability/bland_altman_brightness_age.png")
    bland_altman_plot(reduce(vcat, data[!, "Original Age Prediction"]), reduce(vcat, data[!, "Contrast Age Prediction"]), "Bland-Altman Graf: Originalna vs Procjena dobi s promjenom kontrasta", "stability/bland_altman_contrast_age.png")

    bland_altman_plot(reduce(vcat, data[!, "Original Gender Prediction"]), reduce(vcat, data[!, "Noisy Gender Prediction"]), "Bland-Altman Graf: Originalna vs Procjena spola s bukom", "stability/bland_altman_noisy_gender.png")
    bland_altman_plot(reduce(vcat, data[!, "Original Gender Prediction"]), reduce(vcat, data[!, "Rotation Gender Prediction"]), "Bland-Altman Graf: Originalna vs Procjena spola s rotacijom", "stability/bland_altman_rotation_gender.png")
    bland_altman_plot(reduce(vcat, data[!, "Original Gender Prediction"]), reduce(vcat, data[!, "Brightness Gender Prediction"]), "Bland-Altman Graf: Originalna vs Procjena spola s promjenom osvjetljenja", "stability/bland_altman_brightness_gender.png")
    bland_altman_plot(reduce(vcat, data[!, "Original Gender Prediction"]), reduce(vcat, data[!, "Contrast Gender Prediction"]), "Bland-Altman Graf: Originalna vs Procjena spola s promjenom kontrasta", "stability/bland_altman_contrast_gender.png")
end

# Function to save statistical summary
function save_statistical_summary(data::DataFrame, correlations::Dict{String, Float64}, t_tests::Dict{String, Any}, regressions::Dict{String, StatsModels.TableRegressionModel})
    open("stability/statistical_summary.txt", "w") do f
        println(f, "Osnovna statistika:\n", describe(data))
        
        println(f, "\nKorelacije između originalnih i perturbiranih predikcija:")
        for (k, v) in correlations
            println(f, "$k: $v")
        end

        println(f, "\nT-test rezultati za razlike između originalnih i perturbiranih predikcija:")
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
    file_path = "../results/stability.csv"
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
