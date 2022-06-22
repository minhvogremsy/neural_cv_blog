#include "networkthread.h"

NetworkThread::NetworkThread(int iface_num, QString* group, QStringList* ports) :current_iface(iface_num)
{
    multicast_ports = *ports;
    multicast_group = *group;
}

void NetworkThread::run() {
    QList<QNetworkInterface> list = QNetworkInterface::allInterfaces();
    for (int i = 0; i < list.count(); i++) {
        qDebug() << NETWORK_MODULE_NAME << list.at(i).name() << '\t' << i;
    }
    qDebug() << NETWORK_MODULE_NAME << "used interface " << list.at(current_iface).name();
    Q_FOREACH(QString port, multicast_ports) {
        if (port == MULTICAST_PORT_COMMAND) {
            m_pudp_command = new QUdpSocket(this);
            qDebug() << NETWORK_MODULE_NAME << "m_pudp_arround isBinded =" << m_pudp_command->bind(QHostAddress::AnyIPv4,
                QString(MULTICAST_PORT_COMMAND).toUInt(),
                QUdpSocket::ShareAddress);
            m_pudp_command->setMulticastInterface(list.at(current_iface));
        }
        else if (port == MULTICAST_PORT_HEARTBEAT) {
            m_pudp_heartbeat = new QUdpSocket(this);
            qDebug() << NETWORK_MODULE_NAME << "m_pudp_heartbeat isBinded =" << m_pudp_heartbeat->bind(QHostAddress::AnyIPv4,
                QString(MULTICAST_PORT_HEARTBEAT).toUInt(),
                QUdpSocket::ShareAddress);
            m_pudp_heartbeat->setMulticastInterface(list.at(current_iface));
        }
        else {
            qDebug() << NETWORK_MODULE_NAME << "Unknown port " << port;
        }
    }



    for (int i = 0; i < multicast_ports.count(); ++i) {
        QUdpSocket* m_pudp = new QUdpSocket(this);
        m_pudp->bind(QHostAddress::AnyIPv4, multicast_ports.at(i).toUInt(), QUdpSocket::ShareAddress);
        m_pudp->joinMulticastGroup(QHostAddress(multicast_group));
        connect(m_pudp, SIGNAL(readyRead()), this, SLOT(slotReceiveDatagramm()));
    }
    exec();
}


void NetworkThread::slotSendFile(QStringList file_name) {

}

void NetworkThread::slotSendHeartBeat(QString message)
{
    slotSendDatagramm(m_pudp_heartbeat, QString(MULTICAST_PORT_HEARTBEAT).toUInt(), message.toLocal8Bit());
}

void NetworkThread::slotSendCommand(QString command)
{
    slotSendDatagramm(m_pudp_command, QString(MULTICAST_PORT_COMMAND).toInt(), command.toLocal8Bit());
}

void NetworkThread::slotSendDatagramm(QUdpSocket* socket, int port, QByteArray message)
{
    socket->writeDatagram(message, QHostAddress(multicast_group), port);
    qDebug() << "Sended " << message << multicast_group << port;
}

void NetworkThread::slotReceiveDatagramm() {
    QUdpSocket* m_pudp = reinterpret_cast<QUdpSocket*>(sender());
    while (m_pudp->hasPendingDatagrams()) {
        QByteArray datagram;
        datagram.resize(m_pudp->pendingDatagramSize());
        QHostAddress host;
        quint16 port;

        m_pudp->readDatagram(datagram.data(), datagram.size(), &host, &port);
        qDebug() << NETWORK_MODULE_NAME << datagram;
        if (port == QString(MULTICAST_PORT_COMMAND).toUInt()) {
            command = host.toString() + " " + QString::fromUtf8(datagram);
            emit signalCommand(command);
        }

    }
}
