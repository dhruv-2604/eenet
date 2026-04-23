"""Shared port and peer-ID layout for PRISM P2P network."""

NUM_STAGES = 4
PEERS_PER_STAGE = 4 

STAGE_PORTS = {
    0: [5551, 5552, 5553, 5554],
    1: [5561, 5562, 5563, 5564],
    2: [5571, 5572, 5573, 5574],
    3: [5581, 5582, 5583, 5584],
}

SEGMENT_PATHS = {
    0: "outputs/segments/segment1.pt",
    1: "outputs/segments/segment2.pt",
    2: "outputs/segments/segment3.pt",
    3: "outputs/segments/segment4.pt",
}

# Peers 0-15 Stage 0: 0-3, Stage 1: 4-7, Stage 2: 8-11, Stage 3: 12-15


def peer_id_to_stage(peer_id: int) -> int:
    return peer_id // PEERS_PER_STAGE


def peer_id_to_port(peer_id: int) -> int:
    stage = peer_id_to_stage(peer_id)
    replica = peer_id % PEERS_PER_STAGE
    return STAGE_PORTS[stage][replica]


def peers_by_stage() -> dict:
    return {
        stage: list(range(stage * PEERS_PER_STAGE, (stage + 1) * PEERS_PER_STAGE))
        for stage in range(NUM_STAGES)
    }
