import os
import numpy as np
import pandas as pd
import requests, asyncio, aiohttp
from pmd_beamphysics import ParticleGroup
from datetime import datetime

def filter_by_date(records: list[str], date):
    dates = np.array(list(map(lambda r: datetime.strptime(r[:10], "%Y-%m-%d"), records)))

    return list(np.array(records)[dates < date])


def to_pmd(particles: pd.DataFrame, ref=None) -> ParticleGroup:
    """
    Helper function to transform the particle output from ASTRA to a ParticleGroup object for analysis.

    Parameters
    ----------
    :param particles: DataFrame
        A pandas DataFrame holding information on a particle distribution formatted as defined by ASTRA. Refer
        to the ASTRA manual for further information.
    :return: ParticleGroup
    """
    data = particles.copy()
    ref = ref if ref is not None else data.iloc[0]

    data['weight'] = np.abs(data.pop('macro_charge')) * 1e-9
    data.loc[1:, 'z'] = data.loc[1:, 'z'] + ref['z']
    data.loc[1:, 'pz'] = data.loc[1:, 'pz'] + ref['pz']
    data.loc[1:, 't_clock'] = (data.loc[1:, 't_clock'] + ref['t_clock']) * 1e-9
    data.loc[data['status'] == 1, 'status'] = 2
    data.loc[data['status'] == 5, 'status'] = 1

    data_dict = data.to_dict('list')
    data_dict['n_particles'] = data.size
    data_dict['species'] = 'electron'
    data_dict['t'] = ref['t_clock'] * 1e-9

    return ParticleGroup(data=data_dict)


async def _request(endpoint: str, data: dict, session: aiohttp.ClientSession, request_type: str, request_url: str):
    request_headers = {'Content-Type': 'application/json', 'x-api-key': os.getenv("ASTRA_API_KEY")}
    url = request_url + endpoint
    try:
        request_func = getattr(session, request_type)
        async with request_func(url=url, headers=request_headers, json=data) as response:
            if not response.status == 200:
                print(f"Request [{request_type.upper()}] {url} returned with code {response.status}")
            return await response.json()
    except Exception as e:
        print("Unable to {} {} due to {}.".format(request_type.upper(), url, e.__class__))


async def request(endpoint: str,
                  body: list[dict] | dict,
                  request_type="put",
                  request_url="http://black/astra/",
                  timeout=600,
    ):
    """         The endpoint to be called. Example: "particles"
    :param body: dict, default = {}
        The request body.
    :param request_type: str, default = 'post'
        The request type. One out of ['get', 'post', 'put', 'delete']
    :param request_url: str, default = 'http://black/astra/'
        The url of the server the request is directed to.
    :param timeout: int, default = 600
        Request timeout in seconds
    :return: JSON response
    """
    session_timeout = aiohttp.ClientTimeout(total=None, sock_connect=timeout, sock_read=timeout)
    async with aiohttp.ClientSession(timeout=session_timeout) as session:
        if type(body) is list:
            return (
                await asyncio.gather(*(_request(endpoint, data, session, request_type, request_url) for data in body))
            )
        else:
            return (
                await asyncio.gather(_request(endpoint, body, session, request_type, request_url))
            )[0]

class Simulation:
    def __init__(self, **kwargs):
        self.id = kwargs['sim_id']
        self.input_ini = kwargs['input_ini']
        self.output_ini = kwargs['run_output']
        self.particles = list(map(lambda t: to_pmd(pd.DataFrame(t)), kwargs['particles']))
        self.emittance = {n: pd.DataFrame.from_dict(kwargs[f"emittance_{n}"]) for n in ['x', 'y', 'z']}

    @property
    def final_distribution(self):
        return self.particles[-1]

    def __repr__(self):
        return self.input_ini + "\n\n" + self.output_ini