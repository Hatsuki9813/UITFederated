"""tfflower: A Flower / TensorFlow app."""
import json
from flwr.common import Context, ndarrays_to_parameters, Metrics,EvaluateRes, FitRes, Scalar, parameters_to_ndarrays, Parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from typing import Dict, List, Tuple, Optional, Union
from tfflower.task import load_model
from flwr.server.client_proxy import ClientProxy
from numpy import ndarray, savez
class AggregateCustomMetricStrategy(FedAvg):   
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation accuracy using weighted average."""

        if not results:
            return None, {}

        # Call aggregate_evaluate from base class (FedAvg) to aggregate loss and metrics
        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(
            server_round, results, failures
        )
        with open("aggregated_loss.json", "w") as f:
                json.dump({"aggregated_loss": aggregated_loss}, f)
        print(f" aggregated_loss saved to 'aggregated_loss.json'.")
        # Weigh accuracy of each client by number of examples used
        accuracies = [r.metrics["accuracy"] * r.num_examples for _, r in results]
        examples = [r.num_examples for _, r in results]

        # Aggregate and print custom metric
        aggregated_accuracy = sum(accuracies) / sum(examples)
        print(
            f"Round {server_round} accuracy aggregated from client results: {aggregated_accuracy}"
        )
        if server_round == 1:
            with open("final_accuracy.json", "w") as f:
                json.dump({"final_accuracy": aggregated_accuracy}, f)
            print(f"Final accuracy saved to 'final_accuracy.json'.")
        # Return aggregated loss and metrics (i.e., aggregated accuracy)
        return float(aggregated_loss), {"accuracy": float(aggregated_accuracy)}
    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[Union[tuple[ClientProxy, FitRes], BaseException]],
    ) -> tuple[Optional[Parameters], dict[str, Scalar]]:

        # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )

        if aggregated_parameters is not None:
            # Convert `Parameters` to `list[np.ndarray]`
            aggregated_ndarrays: list[ndarray] = parameters_to_ndarrays(
                aggregated_parameters
            )

            # Save aggregated_ndarrays to disk
            print(f"Saving round {server_round} aggregated_ndarrays...")
            savez(f"round-{server_round}-weights.npz", *aggregated_ndarrays)

        return aggregated_parameters, aggregated_metrics

def on_fit_config(server_round: int) -> Metrics:
    #adjust learning rate based on server round
    lr  = 0.01
    if server_round > 2:
        lr = 0.005
    return {"lr": lr}

def weighted_average(metrics: List[Tuple[int, Metrics]])->Metrics:
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]   
    """Compute the weighted average of a list of metrics."""
    total_examples = sum(num_examples for num_examples, _ in metrics)
    return {
        "accuracy": sum(accuracies) / total_examples,
    }

def server_fn(context: Context):
    # Read from config
    num_rounds = context.run_config["num-server-rounds"]

    # Get parameters to initialize global model
    parameters = ndarrays_to_parameters(load_model().get_weights())

    # Define strategy
    strategy = strategy = AggregateCustomMetricStrategy(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_available_clients=2,
        initial_parameters=parameters,
        evaluate_metrics_aggregation_fn=weighted_average,
        on_fit_config_fn=on_fit_config
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)

# Create ServerApp
app = ServerApp(server_fn=server_fn)
