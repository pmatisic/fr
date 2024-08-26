using CSV
using DataFrames
using Statistics
using Plots
using HypothesisTests
using GLM

# Function to load data
function load_data(file_path::String)::DataFrame
    data = CSV.read(file_path, DataFrame)
    rename!(data, "Test Loss" => :Test_Loss, "Time" => :Time)
    return data
end

# Function to get basic statistics
function get_basic_statistics(data::DataFrame)
    return describe(data)
end

# Function to calculate correlation between Test Loss and Time
function calculate_correlation(data::DataFrame)::Float64
    return cor(data.Test_Loss, data.Time)
end

# Function to perform t-test
function perform_t_test(data::DataFrame)
    return OneSampleTTest(data.Test_Loss, 0.0)
end

# Function to perform linear regression
function perform_linear_regression(data::DataFrame)::StatsModels.TableRegressionModel
    model = lm(@formula(Test_Loss ~ Epoch), data)
    return model
end

# Function to calculate RMSE and MAE
function calculate_errors(model::StatsModels.TableRegressionModel)
    residuals = GLM.residuals(model)
    rmse = sqrt(mean(residuals .^ 2))
    mae = mean(abs.(residuals))
    return rmse, mae
end

# Function to plot Test Loss over Epochs
function plot_test_loss_over_epochs(data::DataFrame)
    plt = plot(data.Epoch, data.Test_Loss, 
               title="Gubitak testiranja kroz epohe", 
               xlabel="Epoha", ylabel="Gubitak testiranja", 
               legend=false)
    save_plot(plt, "test/test_loss_over_epochs.png")
end

# Function to plot Test Loss vs Time
function plot_test_loss_vs_time(data::DataFrame)
    plt = plot(data.Time, data.Test_Loss, 
               title="Gubitak testiranja vs Vrijeme", 
               xlabel="Vrijeme (s)", ylabel="Gubitak testiranja", 
               legend=false)
    save_plot(plt, "test/test_loss_vs_time.png")
end

# Function to plot scatter plot with regression line
function plot_scatter_with_regression(data::DataFrame, model::StatsModels.TableRegressionModel)
    plt = scatter(data.Epoch, data.Test_Loss, 
                  title="Gubitak testiranja vs Epoha", 
                  xlabel="Epoha", ylabel="Gubitak testiranja", 
                  legend=false)
    plot!(plt, data.Epoch, fitted(model), label="Regresijska linija")
    save_plot(plt, "test/scatter_with_regression.png")
end

# Function to plot residuals
function plot_residuals(model::StatsModels.TableRegressionModel)
    residuals = GLM.residuals(model)
    plt = histogram(residuals, bins=30, 
                    title="Distribucija reziduala", 
                    xlabel="Reziduali", ylabel="Frekvencija")
    save_plot(plt, "test/residuals_histogram.png")
end

# Function to save plots
function save_plot(plot, filename::String)
    savefig(plot, filename)
end

# Function to save statistical summary
function save_statistical_summary(data::DataFrame, correlation::Float64, t_test_result, 
                                  linear_model::StatsModels.TableRegressionModel, rmse::Float64, mae::Float64)
    open("test/statistical_summary.txt", "w") do f
        println(f, "Osnovna statistika:\n", describe(data))
        println(f, "\nKorelacija između gubitka testiranja i vremena: ", correlation)
        println(f, "\nT-test rezultat:\n", t_test_result)
        println(f, "\nRezultati linearne regresije:\n", coeftable(linear_model))
        println(f, "\nRMSE: ", rmse)
        println(f, "\nMAE: ", mae)
    end
end

# Main function
function main()
    file_path = "../results/test.csv"
    data = load_data(file_path)
    
    println("Osnovna statistika:\n", get_basic_statistics(data))
    
    correlation = calculate_correlation(data)
    println("Korelacija između gubitka testiranja i vremena: ", correlation)
    
    t_test_result = perform_t_test(data)
    println("T-test rezultat:\n", t_test_result)
    
    linear_model = perform_linear_regression(data)
    println("Rezultati linearne regresije:\n", coeftable(linear_model))
    
    rmse, mae = calculate_errors(linear_model)
    println("RMSE: ", rmse)
    println("MAE: ", mae)
    
    plot_test_loss_over_epochs(data)
    plot_test_loss_vs_time(data)
    plot_scatter_with_regression(data, linear_model)
    plot_residuals(linear_model)
    
    save_statistical_summary(data, correlation, t_test_result, linear_model, rmse, mae)
end

# Execute main function
main()
