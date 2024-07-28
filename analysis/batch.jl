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
               title="Batch Loss over Epochs", 
               xlabel="Epoch", ylabel="Batch Loss", 
               legend=false)
    save_plot(plt, "batch_loss_over_epochs.png")
end

# Function to plot Batch Time over Epochs
function plot_batch_time_over_epochs(data::DataFrame)
    plt = plot(data.Epoch, data.Batch_Time, 
               title="Batch Time over Epochs", 
               xlabel="Epoch", ylabel="Batch Time (s)", 
               legend=false)
    save_plot(plt, "batch_time_over_epochs.png")
end

# Function to plot histogram of Batch Loss
function plot_histogram_batch_loss(data::DataFrame)
    plt = histogram(data.Batch_Loss, bins=30, 
                    title="Histogram of Batch Loss", 
                    xlabel="Batch Loss", ylabel="Frequency")
    save_plot(plt, "batch_loss_histogram.png")
end

# Function to plot histogram of Batch Time
function plot_histogram_batch_time(data::DataFrame)
    plt = histogram(data.Batch_Time, bins=30, 
                    title="Histogram of Batch Time", 
                    xlabel="Batch Time (s)", ylabel="Frequency")
    save_plot(plt, "batch_time_histogram.png")
end

# Function to plot boxplot of Batch Loss per Epoch
function plot_boxplot_batch_loss_per_epoch(data::DataFrame)
    plt = @df data boxplot(:Epoch, :Batch_Loss, 
                           title="Boxplot of Batch Loss per Epoch", 
                           xlabel="Epoch", ylabel="Batch Loss", 
                           legend=false)
    save_plot(plt, "batch_loss_boxplot.png")
end

# Function to calculate correlation between Batch Loss and Batch Time
function calculate_correlation(data::DataFrame)::Float64
    return cor(data.Batch_Loss, data.Batch_Time)
end

# Function to plot scatter plot of Batch Loss vs Batch Time
function plot_scatter_batch_loss_vs_time(data::DataFrame)
    plt = scatter(data.Batch_Loss, data.Batch_Time, 
                  title="Scatter Plot of Batch Loss vs Batch Time", 
                  xlabel="Batch Loss", ylabel="Batch Time (s)", 
                  legend=false)
    save_plot(plt, "batch_loss_vs_time_scatter.png")
end

# Function to perform t-test
function perform_t_test(data::DataFrame, hypothesized_mean::Float64)::OneSampleTTest
    return OneSampleTTest(data.Batch_Loss, hypothesized_mean)
end

# Function to perform ANOVA
function perform_anova(data::DataFrame)
    return fit(LinearModel, @formula(Batch_Loss ~ Epoch), data)
end

# Function to save statistical summary
function save_statistical_summary(data::DataFrame, correlation::Float64, t_test_result::OneSampleTTest, anova_result::StatsModels.TableRegressionModel)
    open("statistical_summary.txt", "w") do f
        println(f, "Basic Statistics:\n", describe(data))
        println(f, "\nCorrelation between Batch Loss and Batch Time: ", correlation)
        println(f, "\nT-test result:\n", t_test_result)
        println(f, "\nANOVA result:\n", anova_result)
    end
end

# Main function
function main()
    file_path = "../results/batch.csv"
    data = load_data(file_path)
    
    println("Basic Statistics:\n", get_basic_statistics(data))
    
    plot_batch_loss_over_epochs(data)
    plot_batch_time_over_epochs(data)
    plot_histogram_batch_loss(data)
    plot_histogram_batch_time(data)
    plot_boxplot_batch_loss_per_epoch(data)
    
    correlation = calculate_correlation(data)
    println("Correlation between Batch Loss and Batch Time: ", correlation)
    plot_scatter_batch_loss_vs_time(data)
    
    t_test_result = perform_t_test(data, 500.0)
    println("T-test result:\n", t_test_result)
    
    anova_result = perform_anova(data)
    println("ANOVA result:\n", anova_result)
    
    save_statistical_summary(data, correlation, t_test_result, anova_result)
end

# Execute main function
main()