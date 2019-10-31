use oscoin_graph_api::types;
use oscoin_graph_api::{Edge, Graph, GraphObject};

pub trait DynamicWeights: Graph {
    fn dynamic_weight(
        &self,
        edge: &impl Edge<Self::Weight, <Self::Node as GraphObject>::Id, Self::EdgeData>,
        hyperparams: &types::HyperParameters<Self::Weight>,
    ) -> Self::Weight;
}
