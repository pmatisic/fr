using CSV
using DataFrames
using Statistics
using Plots
using HypothesisTests
using GLM
using ROCAnalysis
using MLBase

# Function to load data
function load_data(file_path::String)::DataFrame
    data = CSV.read(file_path, DataFrame)
    rename!(data, "Predicted Age" => :Predicted_Age, "Predicted Gender" => :Predicted_Gender,
            "Actual Age" => :Actual_Age, "Actual Gender" => :Actual_Gender)
    return data
end

# Function to get basic statistics
function get_basic_statistics(data::DataFrame)
    return describe(data)
end

# Function to calculate correlation between Predicted Age and Actual Age
function calculate_correlation(data::DataFrame)::Float64
    return cor(data.Predicted_Age, data.Actual_Age)
end

# Function to perform t-test
function perform_t_test(data::DataFrame)
    return UnequalVarianceTTest(data.Predicted_Age, data.Actual_Age)
end

# Function to perform linear regression
function perform_linear_regression(data::DataFrame)::StatsModels.TableRegressionModel
    model = lm(@formula(Actual_Age ~ Predicted_Age), data)
    return model
end

# Function to calculate RMSE and MAE
function calculate_errors(data::DataFrame)
    residuals = data.Predicted_Age - data.Actual_Age
    rmse = sqrt(mean(residuals .^ 2))
    mae = mean(abs.(residuals))
    return rmse, mae
end

# Function to generate confusion matrix for gender prediction
function calculate_confusion_matrix(predicted::Vector{Int}, actual::Vector{Int})
    tp = sum((predicted .== 1) .& (actual .== 1))  # True Positive
    tn = sum((predicted .== 0) .& (actual .== 0))  # True Negative
    fp = sum((predicted .== 1) .& (actual .== 0))  # False Positive
    fn = sum((predicted .== 0) .& (actual .== 1))  # False Negative
    return tp, tn, fp, fn
end

# Function to calculate accuracy, precision, recall, F1-score
function calculate_metrics(tp::Int, tn::Int, fp::Int, fn::Int)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * (precision * recall) / (precision + recall)
    return accuracy, precision, recall, f1_score
end

# Function to calculate ROC AUC
function calculate_roc_auc(data::DataFrame)
    scores = data.Predicted_Gender
    labels = data.Actual_Gender
    roc_curve = ROCAnalysis.roc(scores, labels)
    auc = ROCAnalysis.auc(roc_curve)
    return auc
end

# Function to plot scatter plot with regression line
function plot_scatter_with_regression(data::DataFrame, model::StatsModels.TableRegressionModel)
    plt = scatter(data.Predicted_Age, data.Actual_Age, 
                  title="Predviđena vs Stvarna dob", 
                  xlabel="Predviđena dob", ylabel="Stvarna dob", 
                  legend=false)
    plot!(plt, data.Predicted_Age, fitted(model), label="Regresijska linija")
    save_plot(plt, "predictions/scatter_with_regression.png")
end

# Function to plot residuals
function plot_residuals(data::DataFrame, model::StatsModels.TableRegressionModel)
    residuals = GLM.residuals(model)
    plt = histogram(residuals, bins=30, 
                    title="Distribucija reziduala", 
                    xlabel="Reziduali", ylabel="Frekvencija")
    save_plot(plt, "predictions/residuals_histogram.png")
end

# Function to plot Bland-Altman plot
function plot_bland_altman(data::DataFrame)
    mean_ages = (data.Predicted_Age .+ data.Actual_Age) ./ 2
    diff_ages = data.Predicted_Age .- data.Actual_Age
    mean_diff = mean(diff_ages)
    std_diff = std(diff_ages)
    plt = scatter(mean_ages, diff_ages, 
                  title="Bland-Altman plot", 
                  xlabel="Srednja dob", ylabel="Razlika u dobi", 
                  legend=false)
    hline!(plt, [mean_diff, mean_diff + 1.96*std_diff, mean_diff - 1.96*std_diff], 
           label=["Srednja razlika" "±1.96 SD" "±1.96 SD"], color=[:red :green :green])
    save_plot(plt, "predictions/bland_altman_plot.png")
end

# Function to plot confusion matrix
function plot_confusion_matrix(tp::Int, tn::Int, fp::Int, fn::Int)
    cm = [tn fp; fn tp]
    plt = heatmap(cm, c=:blues, x=["Actual 0", "Actual 1"], y=["Predicted 0", "Predicted 1"],
                  title="Matrica konfuzije", xlabel="Stvarni spol", ylabel="Predviđeni spol", annot=true)
    save_plot(plt, "predictions/confusion_matrix.png")
end

# Function to save plots
function save_plot(plot, filename::String)
    savefig(plot, filename)
end

# Function to save statistical summary
function save_statistical_summary(data::DataFrame, correlation::Float64, t_test_result, 
                                  linear_model::StatsModels.TableRegressionModel, metrics::Tuple, rmse::Float64, mae::Float64, auc::Float64)
    open("predictions/statistical_summary.txt", "w") do f
        println(f, "Osnovna statistika:\n", describe(data))
        println(f, "\nKorelacija između predviđene i stvarne dobi: ", correlation)
        println(f, "\nT-test rezultat:\n", t_test_result)
        println(f, "\nRezultati linearne regresije:\n", coeftable(linear_model))
        println(f, "\nMetrics (accuracy, precision, recall, F1-score):\n", metrics)
        println(f, "\nRMSE: ", rmse)
        println(f, "\nMAE: ", mae)
        println(f, "\nROC AUC: ", auc)
    end
end

# Main function
function main()
    file_path = "../results/predictions.csv"
    data = load_data(file_path)
    
    println("Osnovna statistika:\n", get_basic_statistics(data))
    
    correlation = calculate_correlation(data)
    println("Korelacija između predviđene i stvarne dobi: ", correlation)
    
    t_test_result = perform_t_test(data)
    println("T-test rezultat:\n", t_test_result)
    
    linear_model = perform_linear_regression(data)
    println("Rezultati linearne regresije:\n", coeftable(linear_model))
    
    rmse, mae = calculate_errors(data)
    println("RMSE: ", rmse)
    println("MAE: ", mae)
    
    tp, tn, fp, fn = calculate_confusion_matrix(data.Predicted_Gender, data.Actual_Gender)
    metrics = calculate_metrics(tp, tn, fp, fn)
    println("Metrics (accuracy, precision, recall, F1-score):\n", metrics)
    
    auc = calculate_roc_auc(data)
    println("ROC AUC: ", auc)
    
    plot_residuals(data, linear_model)
    plot_scatter_with_regression(data, linear_model)
    plot_bland_altman(data)
    plot_confusion_matrix(tp, tn, fp, fn)
    
    save_statistical_summary(data, correlation, t_test_result, linear_model, metrics, rmse, mae, auc)
end

# Execute main function
main()
