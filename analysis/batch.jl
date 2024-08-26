using CSV
using DataFrames
using Statistics
using Plots
using HypothesisTests
using GLM
using StatsPlots

# Function to load data
function load_data(file_path::String)::DataFrame
    data = CSV.read(file_path, DataFrame)
    rename!(data, "Batch Loss" => :Batch_Loss, "Batch Time" => :Batch_Time)
    return data
end

# Function to get basic statistics
function get_basic_statistics(data::DataFrame)
    return describe(data)
end

# Function to save plots
function save_plot(plot, filename::String)
    savefig(plot, filename)
end

# Function to plot Batch Loss over Epochs
function plot_batch_loss_over_epochs(data::DataFrame)
    plt = plot(data.Epoch, data.Batch_Loss, 
               title="Gubitak batcha kroz epohe", 
               xlabel="Epoha", ylabel="Gubitak batcha", 
               legend=false)
    save_plot(plt, "batch/batch_loss_over_epochs.png")
end

# Function to plot Batch Time over Epochs
function plot_batch_time_over_epochs(data::DataFrame)
    plt = plot(data.Epoch, data.Batch_Time, 
               title="Vrijeme batcha kroz epohe", 
               xlabel="Epoha", ylabel="Vrijeme batcha (s)", 
               legend=false)
    save_plot(plt, "batch/batch_time_over_epochs.png")
end

# Function to plot histogram of Batch Loss
function plot_histogram_batch_loss(data::DataFrame)
    plt = histogram(data.Batch_Loss, bins=30, 
                    title="Histogram gubitka batcha", 
                    xlabel="Gubitak batcha", ylabel="Frekvencija")
    save_plot(plt, "batch/batch_loss_histogram.png")
end

# Function to plot histogram of Batch Time
function plot_histogram_batch_time(data::DataFrame)
    plt = histogram(data.Batch_Time, bins=30, 
                    title="Histogram vremena batcha", 
                    xlabel="Vrijeme batcha (s)", ylabel="Frekvencija")
    save_plot(plt, "batch/batch_time_histogram.png")
end

# Function to plot boxplot of Batch Loss per Epoch
function plot_boxplot_batch_loss_per_epoch(data::DataFrame)
    plt = @df data boxplot(:Epoch, :Batch_Loss, 
                           title="Boxplot gubitka batcha po epohama", 
                           xlabel="Epoha", ylabel="Gubitak batcha", 
                           legend=false)
    save_plot(plt, "batch/batch_loss_boxplot.png")
end

# Function to calculate correlation between Batch Loss and Batch Time
function calculate_correlation(data::DataFrame)::Float64
    return cor(data.Batch_Loss, data.Batch_Time)
end

# Function to plot scatter plot of Batch Loss vs Batch Time
function plot_scatter_batch_loss_vs_time(data::DataFrame)
    plt = scatter(data.Batch_Loss, data.Batch_Time, 
                  title="Scatter plot gubitka batcha vs vremena batcha", 
                  xlabel="Gubitak batcha", ylabel="Vrijeme batcha (s)", 
                  legend=false)
    save_plot(plt, "batch/batch_loss_vs_time_scatter.png")
end

# Function to perform t-test
function perform_t_test(data::DataFrame, hypothesized_mean::Float64)::OneSampleTTest
    return OneSampleTTest(data.Batch_Loss, hypothesized_mean)
end

# Function to perform linear regression
function perform_linear_regression(data::DataFrame)::StatsModels.TableRegressionModel
    model = lm(@formula(Batch_Loss ~ Epoch), data)
    return model
end

# Function to save statistical summary
function save_statistical_summary(data::DataFrame, correlation::Float64, t_test_result::OneSampleTTest, linear_model::StatsModels.TableRegressionModel)
    open("batch/statistical_summary.txt", "w") do f
        println(f, "Osnovna statistika:\n", describe(data))
        println(f, "\nKorelacija između gubitka batcha i vremena batcha: ", correlation)
        println(f, "\nT-test rezultat:\n", t_test_result)
        println(f, "\nRezultati linearne regresije:\n", coeftable(linear_model))
    end
end

# Function to plot moving average of batch loss
function plot_moving_average(df::DataFrame, window_size::Int)
    moving_avg = [mean(df.Batch_Loss[max(1, i-window_size+1):i]) for i in 1:nrow(df)]
    plt = plot(df.Epoch, moving_avg, xlabel="Epoha", ylabel="Pokretni prosjek gubitka batcha", 
      title="Pokretni prosjek gubitka batcha", legend=false)
    save_plot(plt, "batch/moving_average_batch_loss.png")
end

# Function for outlier detection and plot
function plot_outliers(df::DataFrame)
    threshold = mean(df.Batch_Loss) + 3 * std(df.Batch_Loss)
    outliers = df[df.Batch_Loss .> threshold, :]
    plt = scatter(df.Epoch, df.Batch_Loss, xlabel="Epoha", ylabel="Gubitak batcha", 
             title="Outlieri u gubitku batcha", legend=false)
    scatter!(plt, outliers.Epoch, outliers.Batch_Loss, m=:red)
    save_plot(plt, "batch/batch_loss_outliers.png")
end

# Main function
function main()
    file_path = "../results/batch.csv"
    data = load_data(file_path)
    
    println("Osnovna statistika:\n", get_basic_statistics(data))
    
    plot_batch_loss_over_epochs(data)
    plot_batch_time_over_epochs(data)
    plot_histogram_batch_loss(data)
    plot_histogram_batch_time(data)
    plot_boxplot_batch_loss_per_epoch(data)
    
    correlation = calculate_correlation(data)
    println("Korelacija između gubitka batcha i vremena batcha: ", correlation)
    plot_scatter_batch_loss_vs_time(data)
    t_test_result = perform_t_test(data, 500.0)
    println("T-test rezultat:\n", t_test_result)
    linear_model = perform_linear_regression(data)
    println("Rezultati linearne regresije:\n", coeftable(linear_model))
    
    save_statistical_summary(data, correlation, t_test_result, linear_model)
    
    plot_moving_average(data, 5)
    plot_outliers(data)
end

# Execute main function
main()
